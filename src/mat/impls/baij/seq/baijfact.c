#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: baijfact.c,v 1.58 1998/03/26 22:11:31 balay Exp balay $";
#endif
/*
    Factorization code for BAIJ format. 
*/

#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"


/*
    The symbolic factorization code is identical to that for AIJ format,
  except for very small changes since this is now a SeqBAIJ datastructure.
  NOT good code reuse.
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorSymbolic_SeqBAIJ"
int MatLUFactorSymbolic_SeqBAIJ(Mat A,IS isrow,IS iscol,double f,Mat *B)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data, *b;
  IS          isicol;
  int         *r,*ic, ierr, i, n = a->mbs, *ai = a->i, *aj = a->j;
  int         *ainew,*ajnew, jmax,*fill, *ajtmp, nz, bs = a->bs, bs2=a->bs2;
  int         *idnew, idx, row,m,fm, nnz, nzi,realloc = 0,nzbd,*im;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(isrow,IS_COOKIE);
  PetscValidHeaderSpecific(iscol,IS_COOKIE);
  ierr = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ISGetIndices(isrow,&r); ISGetIndices(isicol,&ic);

  /* get new row pointers */
  ainew = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(ainew);
  ainew[0] = 0;
  /* don't know how many column pointers are needed so estimate */
  jmax = (int) (f*ai[n] + 1);
  ajnew = (int *) PetscMalloc( (jmax)*sizeof(int) ); CHKPTRQ(ajnew);
  /* fill is a linked list of nonzeros in active row */
  fill = (int *) PetscMalloc( (2*n+1)*sizeof(int)); CHKPTRQ(fill);
  im = fill + n + 1;
  /* idnew is location of diagonal in factor */
  idnew = (int *) PetscMalloc( (n+1)*sizeof(int)); CHKPTRQ(idnew);
  idnew[0] = 0;

  for ( i=0; i<n; i++ ) {
    /* first copy previous fill into linked list */
    nnz     = nz    = ai[r[i]+1] - ai[r[i]];
    if (!nz) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,1,"Empty row in matrix");
    ajtmp   = aj + ai[r[i]];
    fill[n] = n;
    while (nz--) {
      fm  = n;
      idx = ic[*ajtmp++];
      do {
        m  = fm;
        fm = fill[m];
      } while (fm < idx);
      fill[m]   = idx;
      fill[idx] = fm;
    }
    row = fill[n];
    while ( row < i ) {
      ajtmp = ajnew + idnew[row] + 1;
      nzbd  = 1 + idnew[row] - ainew[row];
      nz    = im[row] - nzbd;
      fm    = row;
      while (nz-- > 0) {
        idx = *ajtmp++;
        nzbd++;
        if (idx == i) im[row] = nzbd;
        do {
          m  = fm;
          fm = fill[m];
        } while (fm < idx);
        if (fm != idx) {
          fill[m]   = idx;
          fill[idx] = fm;
          fm        = idx;
          nnz++;
        }
      }
      row = fill[row];
    }
    /* copy new filled row into permanent storage */
    ainew[i+1] = ainew[i] + nnz;
    if (ainew[i+1] > jmax) {

      /* estimate how much additional space we will need */
      /* use the strategy suggested by David Hysom <hysom@perch-t.icase.edu> */
      /* just double the memory each time */
      int maxadd = jmax;
      /* maxadd = (int) ((f*(ai[n]+1)*(n-i+5))/n); */
      if (maxadd < nnz) maxadd = (n-i)*(nnz+1);
      jmax += maxadd;

      /* allocate a longer ajnew */
      ajtmp = (int *) PetscMalloc( jmax*sizeof(int) );CHKPTRQ(ajtmp);
      PetscMemcpy(ajtmp,ajnew,ainew[i]*sizeof(int));
      PetscFree(ajnew);
      ajnew = ajtmp;
      realloc++; /* count how many times we realloc */
    }
    ajtmp = ajnew + ainew[i];
    fm    = fill[n];
    nzi   = 0;
    im[i] = nnz;
    while (nnz--) {
      if (fm < i) nzi++;
      *ajtmp++ = fm;
      fm       = fill[fm];
    }
    idnew[i] = ainew[i] + nzi;
  }

  if (ai[n] != 0) {
    double af = ((double)ainew[n])/((double)ai[n]);
    PLogInfo(A,"MatLUFactorSymbolic_SeqBAIJ:Reallocs %d Fill ratio:given %g needed %g\n",
             realloc,f,af);
    PLogInfo(A,"MatLUFactorSymbolic_SeqBAIJ:Run with -pc_lu_fill %g or use \n",af);
    PLogInfo(A,"MatLUFactorSymbolic_SeqBAIJ:PCLUSetFill(pc,%g);\n",af);
    PLogInfo(A,"MatLUFactorSymbolic_SeqBAIJ:for best performance.\n");
  } else {
     PLogInfo(A,"MatLUFactorSymbolic_SeqBAIJ:Empty matrix.\n");
  }

  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);

  PetscFree(fill);

  /* put together the new matrix */
  ierr = MatCreateSeqBAIJ(A->comm,bs,bs*n,bs*n,0,PETSC_NULL,B); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol); 
  b = (Mat_SeqBAIJ *) (*B)->data;
  PetscFree(b->imax);
  b->singlemalloc = 0;
  /* the next line frees the default space generated by the Create() */
  PetscFree(b->a); PetscFree(b->ilen);
  b->a          = (Scalar *) PetscMalloc((ainew[n]+1)*sizeof(Scalar)*bs2);CHKPTRQ(b->a);
  b->j          = ajnew;
  b->i          = ainew;
  b->diag       = idnew;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  b->icol       = isicol;
  b->solve_work = (Scalar *) PetscMalloc( (bs*n+bs)*sizeof(Scalar));CHKPTRQ(b->solve_work);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate idnew, solve_work, new a, new j */
  PLogObjectMemory(*B,(ainew[n]-n)*(sizeof(int)+sizeof(Scalar)));
  b->maxnz = b->nz = ainew[n];
  
  (*B)->info.factor_mallocs    = realloc;
  (*B)->info.fill_ratio_given  = f;
  if (ai[i] != 0) {
    (*B)->info.fill_ratio_needed = ((double)ainew[n])/((double)ai[i]);
  } else {
    (*B)->info.fill_ratio_needed = 0.0;
  }
  

  PetscFunctionReturn(0); 
}

/* ----------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_N"
int MatLUFactorNumeric_SeqBAIJ_N(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS              iscol = b->col, isrow = b->row, isicol = b->icol;
  int             *r,*ic, ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row, bslog,*ai=a->i,*aj=a->j,k,flg;
  int             *diag_offset=b->diag,diag,bs=a->bs,bs2 = a->bs2,*v_pivots;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,*multiplier,*v_work,*pc,*w;
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc(bs2*(n+1)*sizeof(Scalar));CHKPTRQ(rtmp);
  PetscMemzero(rtmp,bs2*(n+1)*sizeof(Scalar));
  /* generate work space needed by dense LU factorization */
  v_work     = (Scalar *) PetscMalloc(bs*sizeof(int) + (bs+bs2)*sizeof(Scalar));
               CHKPTRQ(v_work);
  multiplier = v_work + bs;
  v_pivots   = (int *) (multiplier + bs2);

  /* flops in while loop */
  bslog = 2*bs*bs2;

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) {
      PetscMemzero(rtmp+bs2*ajtmp[j],bs2*sizeof(Scalar));
    }
    /* load in initial (unfactored row) */
    nz       = ai[r[i]+1] - ai[r[i]];
    ajtmpold = aj + ai[r[i]];
    v        = aa + bs2*ai[r[i]];
    for ( j=0; j<nz; j++ ) {
      PetscMemcpy(rtmp+bs2*ic[ajtmpold[j]],v+bs2*j,bs2*sizeof(Scalar)); 
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + bs2*row;
/*      if (*pc) { */
      for ( flg=0,k=0; k<bs2; k++ ) { if (pc[k]!=0.0) { flg =1; break; }}
      if (flg) {
        pv = ba + bs2*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        Kernel_A_gets_A_times_B(bs,pc,pv,multiplier); 
        nz = bi[row+1] - diag_offset[row] - 1;
        pv += bs2;
        for (j=0; j<nz; j++) {
          Kernel_A_gets_A_minus_B_times_C(bs,rtmp+bs2*pj[j],pc,pv+bs2*j);
        }
        PLogFlops(bslog*(nz+1)-bs);
      } 
        row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + bs2*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {
      PetscMemcpy(pv+bs2*j,rtmp+bs2*pj[j],bs2*sizeof(Scalar)); 
    }
    diag = diag_offset[i] - bi[i];
    /* invert diagonal block */
    w = pv + bs2*diag; 
    Kernel_A_gets_inverse_A(bs,w,v_pivots,v_work);
  }

  PetscFree(rtmp); PetscFree(v_work);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(1.3333*bs*bs2*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------*/
/*
      Version for when blocks are 5 by 5
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_5"
int MatLUFactorNumeric_SeqBAIJ_5(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS              iscol = b->col, isrow = b->row, isicol;
  int             *r,*ic, ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row;
  int             *diag_offset = b->diag,idx,*ai=a->i,*aj=a->j;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,*pc,*w,*x;
  Scalar          p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  Scalar          p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  Scalar          x17,x18,x19,x20,x21,x22,x23,x24,x25,p10,p11,p12,p13,p14;
  Scalar          p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,m10,m11,m12;
  Scalar          m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25;
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc(25*(n+1)*sizeof(Scalar));CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) {
      x = rtmp+25*ajtmp[j]; 
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = x[16] = x[17] = 0.0;
      x[18] = x[19] = x[20] = x[21] = x[22] = x[23] = x[24] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 25*ai[idx];
    for ( j=0; j<nz; j++ ) {
      x    = rtmp+25*ic[ajtmpold[j]];
      x[0] = v[0]; x[1] = v[1]; x[2] = v[2]; x[3] = v[3];
      x[4] = v[4]; x[5] = v[5]; x[6] = v[6]; x[7] = v[7]; x[8] = v[8];
      x[9] = v[9]; x[10] = v[10]; x[11] = v[11]; x[12] = v[12]; x[13] = v[13];
      x[14] = v[14]; x[15] = v[15]; x[16] = v[16]; x[17] = v[17]; 
      x[18] = v[18]; x[19] = v[19]; x[20] = v[20]; x[21] = v[21];
      x[22] = v[22]; x[23] = v[23]; x[24] = v[24];
      v    += 25;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 25*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      p5 = pc[4]; p6 = pc[5]; p7 = pc[6]; p8 = pc[7]; p9 = pc[8];
      p10 = pc[9]; p11 = pc[10]; p12 = pc[11]; p13 = pc[12]; p14 = pc[13];
      p15 = pc[14]; p16 = pc[15]; p17 = pc[16]; p18 = pc[17]; p19 = pc[18];
      p20 = pc[19]; p21 = pc[20]; p22 = pc[21]; p23 = pc[22]; p24 = pc[23];
      p25 = pc[24];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 ||
          p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0
          || p16 != 0.0 || p17 != 0.0 || p18 != 0.0 || p19 != 0.0 ||
          p20 != 0.0 || p21 != 0.0 || p22 != 0.0 || p23 != 0.0 || 
          p24 != 0.0 || p25 != 0.0) { 
        pv = ba + 25*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1 = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        x5 = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
        x10 = pv[9]; x11 = pv[10]; x12 = pv[11]; x13 = pv[12]; x14 = pv[13];
        x15 = pv[14]; x16 = pv[15]; x17 = pv[16]; x18 = pv[17]; 
        x19 = pv[18]; x20 = pv[19]; x21 = pv[20]; x22 = pv[21];
        x23 = pv[22]; x24 = pv[23]; x25 = pv[24];
        pc[0] = m1 = p1*x1 + p6*x2  + p11*x3 + p16*x4 + p21*x5;
        pc[1] = m2 = p2*x1 + p7*x2  + p12*x3 + p17*x4 + p22*x5;
        pc[2] = m3 = p3*x1 + p8*x2  + p13*x3 + p18*x4 + p23*x5;
        pc[3] = m4 = p4*x1 + p9*x2  + p14*x3 + p19*x4 + p24*x5;
        pc[4] = m5 = p5*x1 + p10*x2 + p15*x3 + p20*x4 + p25*x5;

        pc[5] = m6 = p1*x6 + p6*x7  + p11*x8 + p16*x9 + p21*x10;
        pc[6] = m7 = p2*x6 + p7*x7  + p12*x8 + p17*x9 + p22*x10;
        pc[7] = m8 = p3*x6 + p8*x7  + p13*x8 + p18*x9 + p23*x10;
        pc[8] = m9 = p4*x6 + p9*x7  + p14*x8 + p19*x9 + p24*x10;
        pc[9] = m10 = p5*x6 + p10*x7 + p15*x8 + p20*x9 + p25*x10;

        pc[10] = m11 = p1*x11 + p6*x12  + p11*x13 + p16*x14 + p21*x15;
        pc[11] = m12 = p2*x11 + p7*x12  + p12*x13 + p17*x14 + p22*x15;
        pc[12] = m13 = p3*x11 + p8*x12  + p13*x13 + p18*x14 + p23*x15;
        pc[13] = m14 = p4*x11 + p9*x12  + p14*x13 + p19*x14 + p24*x15;
        pc[14] = m15 = p5*x11 + p10*x12 + p15*x13 + p20*x14 + p25*x15;

        pc[15] = m16 = p1*x16 + p6*x17  + p11*x18 + p16*x19 + p21*x20;
        pc[16] = m17 = p2*x16 + p7*x17  + p12*x18 + p17*x19 + p22*x20;
        pc[17] = m18 = p3*x16 + p8*x17  + p13*x18 + p18*x19 + p23*x20;
        pc[18] = m19 = p4*x16 + p9*x17  + p14*x18 + p19*x19 + p24*x20;
        pc[19] = m20 = p5*x16 + p10*x17 + p15*x18 + p20*x19 + p25*x20;

        pc[20] = m21 = p1*x21 + p6*x22  + p11*x23 + p16*x24 + p21*x25;
        pc[21] = m22 = p2*x21 + p7*x22  + p12*x23 + p17*x24 + p22*x25;
        pc[22] = m23 = p3*x21 + p8*x22  + p13*x23 + p18*x24 + p23*x25;
        pc[23] = m24 = p4*x21 + p9*x22  + p14*x23 + p19*x24 + p24*x25;
        pc[24] = m25 = p5*x21 + p10*x22 + p15*x23 + p20*x24 + p25*x25;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 25;
        for (j=0; j<nz; j++) {
          x1   = pv[0];  x2 = pv[1];   x3  = pv[2];  x4  = pv[3];
          x5   = pv[4];  x6 = pv[5];   x7  = pv[6];  x8  = pv[7]; x9 = pv[8];
          x10  = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12];
          x14  = pv[13]; x15 = pv[14]; x16 = pv[15]; x17 = pv[16]; 
          x18  = pv[17]; x19 = pv[18]; x20 = pv[19]; x21 = pv[20];
          x22  = pv[21]; x23 = pv[22]; x24 = pv[23]; x25 = pv[24];
          x    = rtmp + 25*pj[j];
          x[0] -= m1*x1 + m6*x2  + m11*x3 + m16*x4 + m21*x5;
          x[1] -= m2*x1 + m7*x2  + m12*x3 + m17*x4 + m22*x5;
          x[2] -= m3*x1 + m8*x2  + m13*x3 + m18*x4 + m23*x5;
          x[3] -= m4*x1 + m9*x2  + m14*x3 + m19*x4 + m24*x5;
          x[4] -= m5*x1 + m10*x2 + m15*x3 + m20*x4 + m25*x5;

          x[5] -= m1*x6 + m6*x7  + m11*x8 + m16*x9 + m21*x10;
          x[6] -= m2*x6 + m7*x7  + m12*x8 + m17*x9 + m22*x10;
          x[7] -= m3*x6 + m8*x7  + m13*x8 + m18*x9 + m23*x10;
          x[8] -= m4*x6 + m9*x7  + m14*x8 + m19*x9 + m24*x10;
          x[9] -= m5*x6 + m10*x7 + m15*x8 + m20*x9 + m25*x10;

          x[10] -= m1*x11 + m6*x12  + m11*x13 + m16*x14 + m21*x15;
          x[11] -= m2*x11 + m7*x12  + m12*x13 + m17*x14 + m22*x15;
          x[12] -= m3*x11 + m8*x12  + m13*x13 + m18*x14 + m23*x15;
          x[13] -= m4*x11 + m9*x12  + m14*x13 + m19*x14 + m24*x15;
          x[14] -= m5*x11 + m10*x12 + m15*x13 + m20*x14 + m25*x15;

          x[15] -= m1*x16 + m6*x17  + m11*x18 + m16*x19 + m21*x20;
          x[16] -= m2*x16 + m7*x17  + m12*x18 + m17*x19 + m22*x20;
          x[17] -= m3*x16 + m8*x17  + m13*x18 + m18*x19 + m23*x20;
          x[18] -= m4*x16 + m9*x17  + m14*x18 + m19*x19 + m24*x20;
          x[19] -= m5*x16 + m10*x17 + m15*x18 + m20*x19 + m25*x20;

          x[20] -= m1*x21 + m6*x22  + m11*x23 + m16*x24 + m21*x25;
          x[21] -= m2*x21 + m7*x22  + m12*x23 + m17*x24 + m22*x25;
          x[22] -= m3*x21 + m8*x22  + m13*x23 + m18*x24 + m23*x25;
          x[23] -= m4*x21 + m9*x22  + m14*x23 + m19*x24 + m24*x25;
          x[24] -= m5*x21 + m10*x22 + m15*x23 + m20*x24 + m25*x25;

          pv   += 25;
        }
        PLogFlops(250*nz+225);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 25*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {
      x     = rtmp+25*pj[j];
      pv[0] = x[0]; pv[1] = x[1]; pv[2] = x[2]; pv[3] = x[3];
      pv[4] = x[4]; pv[5] = x[5]; pv[6] = x[6]; pv[7] = x[7]; pv[8] = x[8];
      pv[9] = x[9]; pv[10] = x[10]; pv[11] = x[11]; pv[12] = x[12];
      pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15]; pv[16] = x[16];
      pv[17] = x[17]; pv[18] = x[18]; pv[19] = x[19]; pv[20] = x[20];
      pv[21] = x[21]; pv[22] = x[22]; pv[23] = x[23]; pv[24] = x[24];
      pv   += 25;
    }
    /* invert diagonal block */
    w = ba + 25*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_5(w); CHKERRQ(ierr);
  }

  PetscFree(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(1.3333*125*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
/*
      Version for when blocks are 4 by 4
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_4"
int MatLUFactorNumeric_SeqBAIJ_4(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS              iscol = b->col, isrow = b->row, isicol;
  int             *r,*ic, ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row;
  int             *diag_offset = b->diag,idx,*ai=a->i,*aj=a->j;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,*pc,*w,*x;
  Scalar          p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  Scalar          p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  Scalar          p10,p11,p12,p13,p14,p15,p16,m10,m11,m12;
  Scalar          m13,m14,m15,m16;
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc(16*(n+1)*sizeof(Scalar));CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) {
      x = rtmp+16*ajtmp[j]; 
      x[0]  = x[1]  = x[2]  = x[3]  = x[4]  = x[5]  = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 16*ai[idx];
    for ( j=0; j<nz; j++ ) {
      x    = rtmp+16*ic[ajtmpold[j]];
      x[0]  = v[0];  x[1]  = v[1];  x[2]  = v[2];  x[3]  = v[3];
      x[4]  = v[4];  x[5]  = v[5];  x[6]  = v[6];  x[7]  = v[7];  x[8]  = v[8];
      x[9]  = v[9];  x[10] = v[10]; x[11] = v[11]; x[12] = v[12]; x[13] = v[13];
      x[14] = v[14]; x[15] = v[15]; 
      v    += 16;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 16*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];  p9  = pc[8];
      p10 = pc[9];  p11 = pc[10]; p12 = pc[11]; p13 = pc[12]; p14 = pc[13];
      p15 = pc[14]; p16 = pc[15]; 
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 ||
          p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0
          || p16 != 0.0) {
        pv = ba + 16*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];  x9  = pv[8];
        x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12]; x14 = pv[13];
        x15 = pv[14]; x16 = pv[15]; 
        pc[0] = m1 = p1*x1 + p5*x2  + p9*x3  + p13*x4;
        pc[1] = m2 = p2*x1 + p6*x2  + p10*x3 + p14*x4;
        pc[2] = m3 = p3*x1 + p7*x2  + p11*x3 + p15*x4;
        pc[3] = m4 = p4*x1 + p8*x2  + p12*x3 + p16*x4;

        pc[4] = m5 = p1*x5 + p5*x6  + p9*x7  + p13*x8;
        pc[5] = m6 = p2*x5 + p6*x6  + p10*x7 + p14*x8;
        pc[6] = m7 = p3*x5 + p7*x6  + p11*x7 + p15*x8;
        pc[7] = m8 = p4*x5 + p8*x6  + p12*x7 + p16*x8;

        pc[8]  = m9  = p1*x9 + p5*x10  + p9*x11  + p13*x12;
        pc[9]  = m10 = p2*x9 + p6*x10  + p10*x11 + p14*x12;
        pc[10] = m11 = p3*x9 + p7*x10  + p11*x11 + p15*x12;
        pc[11] = m12 = p4*x9 + p8*x10  + p12*x11 + p16*x12;

        pc[12] = m13 = p1*x13 + p5*x14  + p9*x15  + p13*x16;
        pc[13] = m14 = p2*x13 + p6*x14  + p10*x15 + p14*x16;
        pc[14] = m15 = p3*x13 + p7*x14  + p11*x15 + p15*x16;
        pc[15] = m16 = p4*x13 + p8*x14  + p12*x15 + p16*x16;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 16;
        for (j=0; j<nz; j++) {
          x1   = pv[0];  x2  = pv[1];   x3 = pv[2];  x4  = pv[3];
          x5   = pv[4];  x6  = pv[5];   x7 = pv[6];  x8  = pv[7]; x9 = pv[8];
          x10  = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12];
          x14  = pv[13]; x15 = pv[14]; x16 = pv[15];
          x    = rtmp + 16*pj[j];
          x[0] -= m1*x1 + m5*x2  + m9*x3  + m13*x4;
          x[1] -= m2*x1 + m6*x2  + m10*x3 + m14*x4;
          x[2] -= m3*x1 + m7*x2  + m11*x3 + m15*x4;
          x[3] -= m4*x1 + m8*x2  + m12*x3 + m16*x4;

          x[4] -= m1*x5 + m5*x6  + m9*x7  + m13*x8;
          x[5] -= m2*x5 + m6*x6  + m10*x7 + m14*x8;
          x[6] -= m3*x5 + m7*x6  + m11*x7 + m15*x8;
          x[7] -= m4*x5 + m8*x6  + m12*x7 + m16*x8;

          x[8]  -= m1*x9 + m5*x10 + m9*x11  + m13*x12;
          x[9]  -= m2*x9 + m6*x10 + m10*x11 + m14*x12;
          x[10] -= m3*x9 + m7*x10 + m11*x11 + m15*x12;
          x[11] -= m4*x9 + m8*x10 + m12*x11 + m16*x12;

          x[12] -= m1*x13 + m5*x14  + m9*x15  + m13*x16;
          x[13] -= m2*x13 + m6*x14  + m10*x15 + m14*x16;
          x[14] -= m3*x13 + m7*x14  + m11*x15 + m15*x16;
          x[15] -= m4*x13 + m8*x14  + m12*x15 + m16*x16;

          pv   += 16;
        }
        PLogFlops(128*nz+112);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 16*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {
      x      = rtmp+16*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; pv[8] = x[8];
      pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11]; pv[12] = x[12];
      pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15];
      pv   += 16;
    }
    /* invert diagonal block */
    w = ba + 16*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_4(w); CHKERRQ(ierr);
  }

  PetscFree(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(1.3333*64*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}
/*
      Version for when blocks are 4 by 4 Using natural ordering
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering"
int MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data,*b = (Mat_SeqBAIJ *)C->data;
  int             ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row;
  int             *diag_offset = b->diag,*ai=a->i,*aj=a->j;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,*pc,*w,*x;
  Scalar          p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  Scalar          p5,p6,p7,p8,p9,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;
  Scalar          p10,p11,p12,p13,p14,p15,p16,m10,m11,m12;
  Scalar          m13,m14,m15,m16;
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  rtmp  = (Scalar *) PetscMalloc(16*(n+1)*sizeof(Scalar));CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) {
      x = rtmp+16*ajtmp[j]; 
      x[0]  = x[1]  = x[2]  = x[3]  = x[4]  = x[5]  = x[6] = x[7] = x[8] = x[9] = 0.0;
      x[10] = x[11] = x[12] = x[13] = x[14] = x[15] = 0.0;
    }
    /* load in initial (unfactored row) */
    nz       = ai[i+1] - ai[i];
    ajtmpold = aj + ai[i];
    v        = aa + 16*ai[i];
    for ( j=0; j<nz; j++ ) {
      x    = rtmp+16*ajtmpold[j];
      x[0]  = v[0];  x[1]  = v[1];  x[2]  = v[2];  x[3]  = v[3];
      x[4]  = v[4];  x[5]  = v[5];  x[6]  = v[6];  x[7]  = v[7];  x[8]  = v[8];
      x[9]  = v[9];  x[10] = v[10]; x[11] = v[11]; x[12] = v[12]; x[13] = v[13];
      x[14] = v[14]; x[15] = v[15]; 
      v    += 16;
    }
    row = *ajtmp++;
    while (row < i) {
      pc  = rtmp + 16*row;
      p1  = pc[0];  p2  = pc[1];  p3  = pc[2];  p4  = pc[3];
      p5  = pc[4];  p6  = pc[5];  p7  = pc[6];  p8  = pc[7];  p9  = pc[8];
      p10 = pc[9];  p11 = pc[10]; p12 = pc[11]; p13 = pc[12]; p14 = pc[13];
      p15 = pc[14]; p16 = pc[15]; 
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0 || p10 != 0.0 ||
          p11 != 0.0 || p12 != 0.0 || p13 != 0.0 || p14 != 0.0 || p15 != 0.0
          || p16 != 0.0) {
        pv = ba + 16*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1  = pv[0];  x2  = pv[1];  x3  = pv[2];  x4  = pv[3];
        x5  = pv[4];  x6  = pv[5];  x7  = pv[6];  x8  = pv[7];  x9  = pv[8];
        x10 = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12]; x14 = pv[13];
        x15 = pv[14]; x16 = pv[15]; 
        pc[0] = m1 = p1*x1 + p5*x2  + p9*x3  + p13*x4;
        pc[1] = m2 = p2*x1 + p6*x2  + p10*x3 + p14*x4;
        pc[2] = m3 = p3*x1 + p7*x2  + p11*x3 + p15*x4;
        pc[3] = m4 = p4*x1 + p8*x2  + p12*x3 + p16*x4;

        pc[4] = m5 = p1*x5 + p5*x6  + p9*x7  + p13*x8;
        pc[5] = m6 = p2*x5 + p6*x6  + p10*x7 + p14*x8;
        pc[6] = m7 = p3*x5 + p7*x6  + p11*x7 + p15*x8;
        pc[7] = m8 = p4*x5 + p8*x6  + p12*x7 + p16*x8;

        pc[8]  = m9  = p1*x9 + p5*x10  + p9*x11  + p13*x12;
        pc[9]  = m10 = p2*x9 + p6*x10  + p10*x11 + p14*x12;
        pc[10] = m11 = p3*x9 + p7*x10  + p11*x11 + p15*x12;
        pc[11] = m12 = p4*x9 + p8*x10  + p12*x11 + p16*x12;

        pc[12] = m13 = p1*x13 + p5*x14  + p9*x15  + p13*x16;
        pc[13] = m14 = p2*x13 + p6*x14  + p10*x15 + p14*x16;
        pc[14] = m15 = p3*x13 + p7*x14  + p11*x15 + p15*x16;
        pc[15] = m16 = p4*x13 + p8*x14  + p12*x15 + p16*x16;

        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 16;
        for (j=0; j<nz; j++) {
          x1   = pv[0];  x2  = pv[1];   x3 = pv[2];  x4  = pv[3];
          x5   = pv[4];  x6  = pv[5];   x7 = pv[6];  x8  = pv[7]; x9 = pv[8];
          x10  = pv[9];  x11 = pv[10]; x12 = pv[11]; x13 = pv[12];
          x14  = pv[13]; x15 = pv[14]; x16 = pv[15];
          x    = rtmp + 16*pj[j];
          x[0] -= m1*x1 + m5*x2  + m9*x3  + m13*x4;
          x[1] -= m2*x1 + m6*x2  + m10*x3 + m14*x4;
          x[2] -= m3*x1 + m7*x2  + m11*x3 + m15*x4;
          x[3] -= m4*x1 + m8*x2  + m12*x3 + m16*x4;

          x[4] -= m1*x5 + m5*x6  + m9*x7  + m13*x8;
          x[5] -= m2*x5 + m6*x6  + m10*x7 + m14*x8;
          x[6] -= m3*x5 + m7*x6  + m11*x7 + m15*x8;
          x[7] -= m4*x5 + m8*x6  + m12*x7 + m16*x8;

          x[8]  -= m1*x9 + m5*x10 + m9*x11  + m13*x12;
          x[9]  -= m2*x9 + m6*x10 + m10*x11 + m14*x12;
          x[10] -= m3*x9 + m7*x10 + m11*x11 + m15*x12;
          x[11] -= m4*x9 + m8*x10 + m12*x11 + m16*x12;

          x[12] -= m1*x13 + m5*x14  + m9*x15  + m13*x16;
          x[13] -= m2*x13 + m6*x14  + m10*x15 + m14*x16;
          x[14] -= m3*x13 + m7*x14  + m11*x15 + m15*x16;
          x[15] -= m4*x13 + m8*x14  + m12*x15 + m16*x16;

          pv   += 16;
        }
        PLogFlops(128*nz+112);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 16*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {
      x      = rtmp+16*pj[j];
      pv[0]  = x[0];  pv[1]  = x[1];  pv[2]  = x[2];  pv[3]  = x[3];
      pv[4]  = x[4];  pv[5]  = x[5];  pv[6]  = x[6];  pv[7]  = x[7]; pv[8] = x[8];
      pv[9]  = x[9];  pv[10] = x[10]; pv[11] = x[11]; pv[12] = x[12];
      pv[13] = x[13]; pv[14] = x[14]; pv[15] = x[15];
      pv   += 16;
    }
    /* invert diagonal block */
    w = ba + 16*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_4(w); CHKERRQ(ierr);
  }

  PetscFree(rtmp);
  C->factor    = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(1.3333*64*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
/*
      Version for when blocks are 3 by 3
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_3"
int MatLUFactorNumeric_SeqBAIJ_3(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS              iscol = b->col, isrow = b->row, isicol;
  int             *r,*ic, ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row, *ai=a->i,*aj=a->j;
  int             *diag_offset = b->diag,idx;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,*pc,*w,*x;
  Scalar          p1,p2,p3,p4,m1,m2,m3,m4,m5,m6,m7,m8,m9,x1,x2,x3,x4;
  Scalar          p5,p6,p7,p8,p9,x5,x6,x7,x8,x9;
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc(9*(n+1)*sizeof(Scalar));CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) {
      x = rtmp + 9*ajtmp[j]; 
      x[0] = x[1] = x[2] = x[3] = x[4] = x[5] = x[6] = x[7] = x[8] = x[9] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 9*ai[idx];
    for ( j=0; j<nz; j++ ) {
      x    = rtmp + 9*ic[ajtmpold[j]];
      x[0] = v[0]; x[1] = v[1]; x[2] = v[2]; x[3] = v[3];
      x[4] = v[4]; x[5] = v[5]; x[6] = v[6]; x[7] = v[7]; x[8] = v[8];
      v    += 9;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 9*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      p5 = pc[4]; p6 = pc[5]; p7 = pc[6]; p8 = pc[7]; p9 = pc[8];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0 || p5 != 0.0 ||
          p6 != 0.0 || p7 != 0.0 || p8 != 0.0 || p9 != 0.0) { 
        pv = ba + 9*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1 = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        x5 = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
        pc[0] = m1 = p1*x1 + p4*x2 + p7*x3;
        pc[1] = m2 = p2*x1 + p5*x2 + p8*x3;
        pc[2] = m3 = p3*x1 + p6*x2 + p9*x3;

        pc[3] = m4 = p1*x4 + p4*x5 + p7*x6;
        pc[4] = m5 = p2*x4 + p5*x5 + p8*x6;
        pc[5] = m6 = p3*x4 + p6*x5 + p9*x6;

        pc[6] = m7 = p1*x7 + p4*x8 + p7*x9;
        pc[7] = m8 = p2*x7 + p5*x8 + p8*x9;
        pc[8] = m9 = p3*x7 + p6*x8 + p9*x9;
        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 9;
        for (j=0; j<nz; j++) {
          x1   = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
          x5   = pv[4]; x6 = pv[5]; x7 = pv[6]; x8 = pv[7]; x9 = pv[8];
          x    = rtmp + 9*pj[j];
          x[0] -= m1*x1 + m4*x2 + m7*x3;
          x[1] -= m2*x1 + m5*x2 + m8*x3;
          x[2] -= m3*x1 + m6*x2 + m9*x3;
 
          x[3] -= m1*x4 + m4*x5 + m7*x6;
          x[4] -= m2*x4 + m5*x5 + m8*x6;
          x[5] -= m3*x4 + m6*x5 + m9*x6;

          x[6] -= m1*x7 + m4*x8 + m7*x9;
          x[7] -= m2*x7 + m5*x8 + m8*x9;
          x[8] -= m3*x7 + m6*x8 + m9*x9;
          pv   += 9;
        }
        PLogFlops(54*nz+36);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 9*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {
      x     = rtmp + 9*pj[j];
      pv[0] = x[0]; pv[1] = x[1]; pv[2] = x[2]; pv[3] = x[3];
      pv[4] = x[4]; pv[5] = x[5]; pv[6] = x[6]; pv[7] = x[7]; pv[8] = x[8];
      pv   += 9;
    }
    /* invert diagonal block */
    w = ba + 9*diag_offset[i];
    ierr = Kernel_A_gets_inverse_A_3(w); CHKERRQ(ierr);
  }

  PetscFree(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(1.3333*27*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
/*
      Version for when blocks are 2 by 2
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_2"
int MatLUFactorNumeric_SeqBAIJ_2(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data,*b = (Mat_SeqBAIJ *)C->data;
  IS              iscol = b->col, isrow = b->row, isicol;
  int             *r,*ic, ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row, v_pivots[2];
  int             *diag_offset=b->diag,bs = 2,idx,*ai=a->i,*aj=a->j;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,m1,m2,m3,m4,*pc,*w,*x,x1,x2,x3,x4;
  Scalar          p1,p2,p3,p4,v_work[2];
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc(4*(n+1)*sizeof(Scalar));CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) {
      x = rtmp+4*ajtmp[j]; x[0] = x[1] = x[2] = x[3] = 0.0;
    }
    /* load in initial (unfactored row) */
    idx      = r[i];
    nz       = ai[idx+1] - ai[idx];
    ajtmpold = aj + ai[idx];
    v        = aa + 4*ai[idx];
    for ( j=0; j<nz; j++ ) {
      x    = rtmp+4*ic[ajtmpold[j]];
      x[0] = v[0]; x[1] = v[1]; x[2] = v[2]; x[3] = v[3];
      v    += 4;
    }
    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + 4*row;
      p1 = pc[0]; p2 = pc[1]; p3 = pc[2]; p4 = pc[3];
      if (p1 != 0.0 || p2 != 0.0 || p3 != 0.0 || p4 != 0.0) { 
        pv = ba + 4*diag_offset[row];
        pj = bj + diag_offset[row] + 1;
        x1 = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
        pc[0] = m1 = p1*x1 + p3*x2;
        pc[1] = m2 = p2*x1 + p4*x2;
        pc[2] = m3 = p1*x3 + p3*x4;
        pc[3] = m4 = p2*x3 + p4*x4;
        nz = bi[row+1] - diag_offset[row] - 1;
        pv += 4;
        for (j=0; j<nz; j++) {
          x1   = pv[0]; x2 = pv[1]; x3 = pv[2]; x4 = pv[3];
          x    = rtmp + 4*pj[j];
          x[0] -= m1*x1 + m3*x2;
          x[1] -= m2*x1 + m4*x2;
          x[2] -= m1*x3 + m3*x4;
          x[3] -= m2*x3 + m4*x4;
          pv   += 4;
        }
        PLogFlops(16*nz+12);
      } 
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + 4*bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {
      x     = rtmp+4*pj[j];
      pv[0] = x[0]; pv[1] = x[1]; pv[2] = x[2]; pv[3] = x[3];
      pv   += 4;
    }
    /* invert diagonal block */
    w = ba + 4*diag_offset[i];
    Kernel_A_gets_inverse_A(bs,w,v_pivots,v_work);
  }

  PetscFree(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(1.3333*8*b->mbs); /* from inverting diagonal blocks */
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
/*
     Version for when blocks are 1 by 1.
*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqBAIJ_1"
int MatLUFactorNumeric_SeqBAIJ_1(Mat A,Mat *B)
{
  Mat             C = *B;
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data, *b = (Mat_SeqBAIJ *)C->data;
  IS              iscol = b->col, isrow = b->row, isicol;
  int             *r,*ic, ierr, i, j, n = a->mbs, *bi = b->i, *bj = b->j;
  int             *ajtmpold, *ajtmp, nz, row,*ai = a->i,*aj = a->j;
  int             *diag_offset = b->diag,diag;
  register int    *pj;
  register Scalar *pv,*v,*rtmp,multiplier,*pc;
  Scalar          *ba = b->a,*aa = a->a;

  PetscFunctionBegin;
  ierr  = ISInvertPermutation(iscol,&isicol); CHKERRQ(ierr);
  PLogObjectParent(*B,isicol);
  ierr  = ISGetIndices(isrow,&r); CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic); CHKERRQ(ierr);
  rtmp  = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar));CHKPTRQ(rtmp);

  for ( i=0; i<n; i++ ) {
    nz    = bi[i+1] - bi[i];
    ajtmp = bj + bi[i];
    for  ( j=0; j<nz; j++ ) rtmp[ajtmp[j]] = 0.0;

    /* load in initial (unfactored row) */
    nz       = ai[r[i]+1] - ai[r[i]];
    ajtmpold = aj + ai[r[i]];
    v        = aa + ai[r[i]];
    for ( j=0; j<nz; j++ ) rtmp[ic[ajtmpold[j]]] =  v[j];

    row = *ajtmp++;
    while (row < i) {
      pc = rtmp + row;
      if (*pc != 0.0) {
        pv         = ba + diag_offset[row];
        pj         = bj + diag_offset[row] + 1;
        multiplier = *pc * *pv++;
        *pc        = multiplier;
        nz         = bi[row+1] - diag_offset[row] - 1;
        for (j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
        PLogFlops(1+2*nz);
      }
      row = *ajtmp++;
    }
    /* finished row so stick it into b->a */
    pv = ba + bi[i];
    pj = bj + bi[i];
    nz = bi[i+1] - bi[i];
    for ( j=0; j<nz; j++ ) {pv[j] = rtmp[pj[j]];}
    diag = diag_offset[i] - bi[i];
    /* check pivot entry for current row */
    if (pv[diag] == 0.0) {
      SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,0,"Zero pivot");
    }
    pv[diag] = 1.0/pv[diag];
  }

  PetscFree(rtmp);
  ierr = ISRestoreIndices(isicol,&ic); CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r); CHKERRQ(ierr);
  ierr = ISDestroy(isicol); CHKERRQ(ierr);
  C->factor    = FACTOR_LU;
  C->assembled = PETSC_TRUE;
  PLogFlops(b->n);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MatLUFactor_SeqBAIJ"
int MatLUFactor_SeqBAIJ(Mat A,IS row,IS col,double f)
{
  Mat_SeqBAIJ    *mat = (Mat_SeqBAIJ *) A->data;
  int            ierr;
  Mat            C;
  PetscOps       *Abops;
  struct _MatOps *Aops;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic(A,row,col,f,&C); CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(A,&C); CHKERRQ(ierr);

  /* free all the data structures from mat */
  PetscFree(mat->a); 
  if (!mat->singlemalloc) {PetscFree(mat->i); PetscFree(mat->j);}
  if (mat->diag) PetscFree(mat->diag);
  if (mat->ilen) PetscFree(mat->ilen);
  if (mat->imax) PetscFree(mat->imax);
  if (mat->solve_work) PetscFree(mat->solve_work);
  if (mat->mult_work) PetscFree(mat->mult_work);
  PetscFree(mat);

  /*
       This is horrible, horrible code. We need to keep the 
    A pointers for the bops and ops but copy everything 
    else from C.
  */
  Abops = A->bops;
  Aops  = A->ops;
  PetscMemcpy(A,C,sizeof(struct _p_Mat));
  A->bops = Abops;
  A->ops  = Aops;

  PetscHeaderDestroy(C);
  PetscFunctionReturn(0);
}
