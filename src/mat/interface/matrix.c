#ifndef lint
static char vcid[] = "$Id: matrix.c,v 1.181 1996/07/21 16:39:08 bsmith Exp bsmith $";
#endif

/*
   This is where the abstract matrix operations are defined
*/

#include "petsc.h"
#include "matimpl.h"        /*I "mat.h" I*/
#include "src/vec/vecimpl.h"  
#include "pinclude/pviewer.h"
#include "draw.h"
       
/*@C
   MatGetReordering - Gets a reordering for a matrix to reduce fill or to
   improve numerical stability of LU factorization.

   Input Parameters:
.  mat - the matrix
.  type - type of reordering, one of the following:
$      ORDER_NATURAL - Natural
$      ORDER_ND - Nested Dissection
$      ORDER_1WD - One-way Dissection
$      ORDER_RCM - Reverse Cuthill-McGee
$      ORDER_QMD - Quotient Minimum Degree

   Output Parameters:
.  rperm - row permutation indices
.  cperm - column permutation indices

   Options Database Keys:
   To specify the ordering through the options database, use one of
   the following 
$    -mat_order natural, -mat_order nd, -mat_order 1wd, 
$    -mat_order rcm, -mat_order qmd

   Notes:
   If the column permutations and row permutations are the same, 
   then MatGetReordering() returns 0 in cperm.

   The user can define additional orderings; see MatReorderingRegister().

.keywords: matrix, set, ordering, factorization, direct, ILU, LU,
           fill, reordering, natural, Nested Dissection,
           One-way Dissection, Cholesky, Reverse Cuthill-McGee, 
           Quotient Minimum Degree

.seealso:  MatGetReorderingTypeFromOptions(), MatReorderingRegister()
@*/
int MatGetReordering(Mat mat,MatReordering type,IS *rperm,IS *cperm)
{
  int         ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatGetReordering:Not for unassembled matrix");

  if (!mat->ops.getreordering) {*rperm = 0; *cperm = 0; return 0;}
  PLogEventBegin(MAT_GetReordering,mat,0,0,0);
  ierr = MatGetReorderingTypeFromOptions(0,&type); CHKERRQ(ierr);
  ierr = (*mat->ops.getreordering)(mat,type,rperm,cperm); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetReordering,mat,0,0,0);
  return 0;
}

/*@C
   MatGetRow - Gets a row of a matrix.  You MUST call MatRestoreRow()
   for each row that you get to ensure that your application does
   not bleed memory.

   Input Parameters:
.  mat - the matrix
.  row - the row to get

   Output Parameters:
.  ncols -  the number of nonzeros in the row
.  cols - if nonzero, the column numbers
.  vals - if nonzero, the values

   Notes:
   This routine is provided for people who need to have direct access
   to the structure of a matrix.  We hope that we provide enough
   high-level matrix routines that few users will need it. 

   For better efficiency, set cols and/or vals to PETSC_NULL if you do
   not wish to extract these quantities.

   The user can only examine the values extracted with MatGetRow();
   the values cannot be altered.  To change the matrix entries, one
   must use MatSetValues().

   Caution:
   Do not try to change the contents of the output arrays (cols and vals).
   In some cases, this may corrupt the matrix.

.keywords: matrix, row, get, extract

.seealso: MatRestoreRow(), MatSetValues()
@*/
int MatGetRow(Mat mat,int row,int *ncols,int **cols,Scalar **vals)
{
  int   ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(ncols);
  if (!mat->assembled) SETERRQ(1,"MatGetRow:Not for unassembled matrix");
  PLogEventBegin(MAT_GetRow,mat,0,0,0);
  ierr = (*mat->ops.getrow)(mat,row,ncols,cols,vals); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetRow,mat,0,0,0);
  return 0;
}

/*@C  
   MatRestoreRow - Frees any temporary space allocated by MatGetRow().

   Input Parameters:
.  mat - the matrix
.  row - the row to get
.  ncols, cols - the number of nonzeros and their columns
.  vals - if nonzero the column values

.keywords: matrix, row, restore

.seealso:  MatGetRow()
@*/
int MatRestoreRow(Mat mat,int row,int *ncols,int **cols,Scalar **vals)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(ncols);
  if (!mat->assembled) SETERRQ(1,"MatRestoreRow:Not for unassembled matrix");
  if (!mat->ops.restorerow) return 0;
  return (*mat->ops.restorerow)(mat,row,ncols,cols,vals);
}
/*@
   MatView - Visualizes a matrix object.

   Input Parameters:
.  mat - the matrix
.  ptr - visualization context

   Notes:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file
$    ViewerFileOpenBinary() - output in binary to a
$         specified file; corresponding input uses MatLoad()
$    ViewerDrawOpenX() - output nonzero matrix structure to 
$         an X window display
$    ViewerMatlabOpen() - output matrix to Matlab viewer.
$         Currently only the sequential dense and AIJ
$         matrix types support the Matlab viewer.

   The user can call ViewerSetFormat() to specify the output
   format of ASCII printed objects (when using VIEWER_STDOUT_SELF,
   VIEWER_STDOUT_WORLD and ViewerFileOpenASCII).  Available formats include
$    ASCII_FORMAT_DEFAULT - default, prints matrix contents
$    ASCII_FORMAT_MATLAB - Matlab format
$    ASCII_FORMAT_IMPL - implementation-specific format
$      (which is in many cases the same as the default)
$    ASCII_FORMAT_INFO - basic information about the matrix
$      size and structure (not the matrix entries)
$    ASCII_FORMAT_INFO_DETAILED - more detailed information about the 
$      matrix structure

.keywords: matrix, view, visualize, output, print, write, draw

.seealso: ViewerSetFormat(), ViewerFileOpenASCII(), ViewerDrawOpenX(), 
          ViewerMatlabOpen(), ViewerFileOpenBinary(), MatLoad()
@*/
int MatView(Mat mat,Viewer viewer)
{
  int          format, ierr, rows, cols,nz, nzalloc, mem;
  FILE         *fd;
  char         *cstr;
  ViewerType   vtype;
  MPI_Comm     comm = mat->comm;

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatView:Not for unassembled matrix");

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF;
  }

  ierr = ViewerGetType(viewer,&vtype);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);  
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (format == ASCII_FORMAT_INFO || format == ASCII_FORMAT_INFO_DETAILED) {
      PetscFPrintf(comm,fd,"Matrix Object:\n");
      ierr = MatGetType(mat,PETSC_NULL,&cstr); CHKERRQ(ierr);
      ierr = MatGetSize(mat,&rows,&cols); CHKERRQ(ierr);
      PetscFPrintf(comm,fd,"  type=%s, rows=%d, cols=%d\n",cstr,rows,cols);
      if (mat->ops.getinfo) {
        ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&nz,&nzalloc,&mem); CHKERRQ(ierr);
        PetscFPrintf(comm,fd,"  total: nonzeros=%d, allocated nonzeros=%d\n",nz,
                     nzalloc);
      }
    }
  }
  if (mat->view) {ierr = (*mat->view)((PetscObject)mat,viewer); CHKERRQ(ierr);}
  return 0;
}

/*@C
   MatDestroy - Frees space taken by a matrix.
  
   Input Parameter:
.  mat - the matrix

.keywords: matrix, destroy
@*/
int MatDestroy(Mat mat)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = (*mat->destroy)((PetscObject)mat); CHKERRQ(ierr);
  return 0;
}
/*@
   MatValid - Checks whether a matrix object is valid.

   Input Parameter:
.  m - the matrix to check 

   Output Parameter:
   flg - flag indicating matrix status, either
$     PETSC_TRUE if matrix is valid;
$     PETSC_FALSE otherwise.

.keywords: matrix, valid
@*/
int MatValid(Mat m,PetscTruth *flg)
{
  PetscValidIntPointer(flg);
  if (!m)                           *flg = PETSC_FALSE;
  else if (m->cookie != MAT_COOKIE) *flg = PETSC_FALSE;
  else                              *flg = PETSC_TRUE;
  return 0;
}

/*@ 
   MatSetValues - Inserts or adds a block of values into a matrix.
   These values may be cached, so MatAssemblyBegin() and MatAssemblyEnd() 
   MUST be called after all calls to MatSetValues() have been completed.

   Input Parameters:
.  mat - the matrix
.  v - a logically two-dimensional array of values
.  m, indexm - the number of rows and their global indices 
.  n, indexn - the number of columns and their global indices
.  addv - either ADD_VALUES or INSERT_VALUES, where
$     ADD_VALUES - adds values to any existing entries
$     INSERT_VALUES - replaces existing entries with new values

   Notes:
   By default the values, v, are row-oriented and unsorted.
   See MatSetOptions() for other options.

   Calls to MatSetValues() with the INSERT_VALUES and ADD_VALUES 
   options cannot be mixed without intervening calls to the assembly
   routines.

.keywords: matrix, insert, add, set, values

.seealso: MatSetOptions(), MatAssemblyBegin(), MatAssemblyEnd()
@*/
int MatSetValues(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v,InsertMode addv)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!m || !n) return 0; /* no values to insert */
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);

  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  PLogEventBegin(MAT_SetValues,mat,0,0,0);
  ierr = (*mat->ops.setvalues)(mat,m,idxm,n,idxn,v,addv);CHKERRQ(ierr);
  PLogEventEnd(MAT_SetValues,mat,0,0,0);  
  return 0;
}

/*@ 
   MatGetValues - Gets a block of values from a matrix.

   Input Parameters:
.  mat - the matrix
.  v - a logically two-dimensional array for storing the values
.  m, indexm - the number of rows and their global indices 
.  n, indexn - the number of columns and their global indices

   Notes:
   The user must allocate space (m*n Scalars) for the values, v.
   The values, v, are then returned in a row-oriented format, 
   analogous to that used by default in MatSetValues().

.keywords: matrix, get, values

.seealso: MatGetRow(), MatGetSubmatrices(), MatSetValues()
@*/
int MatGetValues(Mat mat,int m,int *idxm,int n,int *idxn,Scalar *v)
{
  int ierr;

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(idxm);
  PetscValidIntPointer(idxn);
  PetscValidScalarPointer(v);
  if (!mat->assembled) SETERRQ(1,"MatGetValues:Not for unassembled matrix");

  PLogEventBegin(MAT_GetValues,mat,0,0,0);
  ierr = (*mat->ops.getvalues)(mat,m,idxm,n,idxn,v); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetValues,mat,0,0,0);
  return 0;
}

/* --------------------------------------------------------*/
/*@
   MatMult - Computes the matrix-vector product, y = Ax.

   Input Parameters:
.  mat - the matrix
.  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

.keywords: matrix, multiply, matrix-vector product

.seealso: MatMultTrans(), MatMultAdd(), MatMultTransAdd()
@*/
int MatMult(Mat mat,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE); 
  if (!mat->assembled) SETERRQ(1,"MatMult:Not for unassembled matrix");
  /* if (mat->factor) SETERRQ(1,"MatMult:Not for factored matrix"); */
  if (x == y) SETERRQ(1,"MatMult:x and y must be different vectors");
  if (mat->N != x->N) SETERRQ(PETSC_ERR_SIZ,"MatMult:Mat mat,Vec x: global dim"); 
  if (mat->M != y->N) SETERRQ(PETSC_ERR_SIZ,"MatMult:Mat mat,Vec y: global dim"); 
  if (mat->m != y->n) SETERRQ(PETSC_ERR_SIZ,"MatMult:Mat mat,Vec y: local dim"); 

  PLogEventBegin(MAT_Mult,mat,x,y,0);
  ierr = (*mat->ops.mult)(mat,x,y); CHKERRQ(ierr);
  PLogEventEnd(MAT_Mult,mat,x,y,0);
  return 0;
}   
/*@
   MatMultTrans - Computes matrix transpose times a vector.

   Input Parameters:
.  mat - the matrix
.  x   - the vector to be multilplied

   Output Parameters:
.  y - the result

.keywords: matrix, multiply, matrix-vector product, transpose

.seealso: MatMult(), MatMultAdd(), MatMultTransAdd()
@*/
int MatMultTrans(Mat mat,Vec x,Vec y)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatMultTrans:Not for unassembled matrix");
  /* if (mat->factor) SETERRQ(1,"MatMult:Not for factored matrix"); */
  if (x == y) SETERRQ(1,"MatMultTrans:x and y must be different vectors");
  if (mat->M != x->N) SETERRQ(PETSC_ERR_SIZ,"MatMultTrans:Mat mat,Vec x: global dim"); 
  if (mat->N != y->N) SETERRQ(PETSC_ERR_SIZ,"MatMultTrans:Mat mat,Vec y: global dim"); 

  PLogEventBegin(MAT_MultTrans,mat,x,y,0);
  ierr = (*mat->ops.multtrans)(mat,x,y); CHKERRQ(ierr);
  PLogEventEnd(MAT_MultTrans,mat,x,y,0);
  return 0;
}   
/*@
    MatMultAdd -  Computes v3 = v2 + A * v1.

    Input Parameters:
.   mat - the matrix
.   v1, v2 - the vectors

    Output Parameters:
.   v3 - the result

.keywords: matrix, multiply, matrix-vector product, add

.seealso: MatMultTrans(), MatMult(), MatMultTransAdd()
@*/
int MatMultAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(v1,VEC_COOKIE);
  PetscValidHeaderSpecific(v2,VEC_COOKIE); PetscValidHeaderSpecific(v3,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatMultAdd:Not for unassembled matrix");
  /* if (mat->factor) SETERRQ(1,"MatMult:Not for factored matrix"); */
  if (mat->N != v1->N) SETERRQ(PETSC_ERR_SIZ,"MatMultAdd:Mat mat,Vec v1: global dim");
  if (mat->M != v2->N) SETERRQ(PETSC_ERR_SIZ,"MatMultAdd:Mat mat,Vec v2: global dim");
  if (mat->M != v3->N) SETERRQ(PETSC_ERR_SIZ,"MatMultAdd:Mat mat,Vec v3: global dim");
  if (mat->m != v3->n) SETERRQ(PETSC_ERR_SIZ,"MatMultAdd:Mat mat,Vec v3: local dim"); 
  if (mat->m != v2->n) SETERRQ(PETSC_ERR_SIZ,"MatMultAdd:Mat mat,Vec v2: local dim"); 

  PLogEventBegin(MAT_MultAdd,mat,v1,v2,v3);
  if (v1 == v3) SETERRQ(1,"MatMultAdd:v1 and v3 must be different vectors");
  ierr = (*mat->ops.multadd)(mat,v1,v2,v3); CHKERRQ(ierr);
  PLogEventEnd(MAT_MultAdd,mat,v1,v2,v3);
  return 0;
}   
/*@
   MatMultTransAdd - Computes v3 = v2 + A' * v1.

   Input Parameters:
.  mat - the matrix
.  v1, v2 - the vectors

   Output Parameters:
.  v3 - the result

.keywords: matrix, multiply, matrix-vector product, transpose, add

.seealso: MatMultTrans(), MatMultAdd(), MatMult()
@*/
int MatMultTransAdd(Mat mat,Vec v1,Vec v2,Vec v3)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(v1,VEC_COOKIE);
  PetscValidHeaderSpecific(v2,VEC_COOKIE);PetscValidHeaderSpecific(v3,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatMultTransAdd:Not for unassembled matrix");
  /* if (mat->factor) SETERRQ(1,"MatMult:Not for factored matrix"); */
  if (!mat->ops.multtransadd) SETERRQ(PETSC_ERR_SUP,"MatMultTransAdd");
  if (v1 == v3) SETERRQ(1,"MatMultTransAdd:v1 and v2 must be different vectors");
  if (mat->M != v1->N) SETERRQ(PETSC_ERR_SIZ,"MatMultTransAdd:Mat mat,Vec v1: global dim");
  if (mat->N != v2->N) SETERRQ(PETSC_ERR_SIZ,"MatMultTransAdd:Mat mat,Vec v2: global dim");
  if (mat->N != v3->N) SETERRQ(PETSC_ERR_SIZ,"MatMultTransAdd:Mat mat,Vec v3: global dim");

  PLogEventBegin(MAT_MultTransAdd,mat,v1,v2,v3);
  ierr = (*mat->ops.multtransadd)(mat,v1,v2,v3); CHKERRQ(ierr);
  PLogEventEnd(MAT_MultTransAdd,mat,v1,v2,v3); 
  return 0;
}
/* ------------------------------------------------------------*/
/*@C
   MatGetInfo - Returns information about matrix storage (number of
   nonzeros, memory).

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  flag - flag indicating the type of parameters to be returned
$    flag = MAT_LOCAL: local matrix
$    flag = MAT_GLOBAL_MAX: maximum over all processors
$    flag = MAT_GLOBAL_SUM: sum over all processors
.   nz - the number of nonzeros [or PETSC_NULL]
.   nzalloc - the number of allocated nonzeros [or PETSC_NULL]
.   mem - the memory used (in bytes)  [or PETSC_NULL]

.keywords: matrix, get, info, storage, nonzeros, memory
@*/
int MatGetInfo(Mat mat,MatInfoType flag,int *nz,int *nzalloc,int *mem)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.getinfo) SETERRQ(PETSC_ERR_SUP,"MatGetInfo");
  return  (*mat->ops.getinfo)(mat,flag,nz,nzalloc,mem);
}   
/* ----------------------------------------------------------*/
/*@  
   MatILUDTFactor - Performs a drop tolerance ILU factorization.

   Input Parameters:
.  mat - the matrix
.  dt  - the drop tolerance
.  maxnz - the maximum number of nonzeros per row allowed?
.  row - row permutation
.  col - column permutation

   Output Parameters:
.  fact - the factored matrix

.keywords: matrix, factor, LU, in-place

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatILUDTFactor(Mat mat,double dt,int maxnz,IS row,IS col,Mat *fact)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.iludtfactor) SETERRQ(PETSC_ERR_SUP,"MatILUDTFactor");
  if (!mat->assembled) SETERRQ(1,"MatILUDTFactor:Not for unassembled matrix");

  PLogEventBegin(MAT_ILUFactor,mat,row,col,0); 
  ierr = (*mat->ops.iludtfactor)(mat,dt,maxnz,row,col,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_ILUFactor,mat,row,col,0);

  return 0;
}

/*@  
   MatLUFactor - Performs in-place LU factorization of matrix.

   Input Parameters:
.  mat - the matrix
.  row - row permutation
.  col - column permutation
.  f - expected fill as ratio of original fill.

.keywords: matrix, factor, LU, in-place

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatLUFactor(Mat mat,IS row,IS col,double f)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.lufactor) SETERRQ(PETSC_ERR_SUP,"MatLUFactor");
  if (!mat->assembled) SETERRQ(1,"MatLUFactor:Not for unassembled matrix");

  PLogEventBegin(MAT_LUFactor,mat,row,col,0); 
  ierr = (*mat->ops.lufactor)(mat,row,col,f); CHKERRQ(ierr);
  PLogEventEnd(MAT_LUFactor,mat,row,col,0); 
  return 0;
}
/*@  
   MatILUFactor - Performs in-place ILU factorization of matrix.

   Input Parameters:
.  mat - the matrix
.  row - row permutation
.  col - column permutation
.  f - expected fill as ratio of original fill.
.  level - number of levels of fill.

   Note: probably really only in-place when level is zero.
.keywords: matrix, factor, ILU, in-place

.seealso: MatILUFactorSymbolic(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatILUFactor(Mat mat,IS row,IS col,double f,int level)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.ilufactor) SETERRQ(PETSC_ERR_SUP,"MatILUFactor");
  if (!mat->assembled) SETERRQ(1,"MatILUFactor:Not for unassembled matrix");

  PLogEventBegin(MAT_ILUFactor,mat,row,col,0); 
  ierr = (*mat->ops.ilufactor)(mat,row,col,f,level); CHKERRQ(ierr);
  PLogEventEnd(MAT_ILUFactor,mat,row,col,0); 
  return 0;
}

/*@  
   MatLUFactorSymbolic - Performs symbolic LU factorization of matrix.
   Call this routine before calling MatLUFactorNumeric().

   Input Parameters:
.  mat - the matrix
.  row, col - row and column permutations
.  f - expected fill as ratio of the original number of nonzeros, 
       for example 3.0; choosing this parameter well can result in 
       more efficient use of time and space.

   Output Parameter:
.  fact - new matrix that has been symbolically factored

   Options Database Key:
$     -mat_lu_fill <f>, where f is the fill ratio

   Notes:
   See the file $(PETSC_DIR)/Performace for additional information about
   choosing the fill factor for better efficiency.

.keywords: matrix, factor, LU, symbolic, fill

.seealso: MatLUFactor(), MatLUFactorNumeric(), MatCholeskyFactor()
@*/
int MatLUFactorSymbolic(Mat mat,IS row,IS col,double f,Mat *fact)
{
  int ierr,flg;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!fact) SETERRQ(1,"MatLUFactorSymbolic:Missing factor matrix argument");
  if (!mat->ops.lufactorsymbolic) SETERRQ(PETSC_ERR_SUP,"MatLUFactorSymbolic");
  if (!mat->assembled) SETERRQ(1,"MatLUFactorSymbolic:Not for unassembled matrix");

  ierr = OptionsGetDouble(PETSC_NULL,"-mat_lu_fill",&f,&flg); CHKERRQ(ierr);
  PLogEventBegin(MAT_LUFactorSymbolic,mat,row,col,0); 
  ierr = (*mat->ops.lufactorsymbolic)(mat,row,col,f,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_LUFactorSymbolic,mat,row,col,0); 
  return 0;
}
/*@  
   MatLUFactorNumeric - Performs numeric LU factorization of a matrix.
   Call this routine after first calling MatLUFactorSymbolic().

   Input Parameters:
.  mat - the matrix
.  row, col - row and  column permutations

   Output Parameters:
.  fact - symbolically factored matrix that must have been generated
          by MatLUFactorSymbolic()

   Notes:
   See MatLUFactor() for in-place factorization.  See 
   MatCholeskyFactorNumeric() for the symmetric, positive definite case.  

.keywords: matrix, factor, LU, numeric

.seealso: MatLUFactorSymbolic(), MatLUFactor(), MatCholeskyFactor()
@*/
int MatLUFactorNumeric(Mat mat,Mat *fact)
{
  int ierr,flg;

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!fact) SETERRQ(1,"MatLUFactorNumeric:Missing factor matrix argument");
  if (!mat->ops.lufactornumeric) SETERRQ(PETSC_ERR_SUP,"MatLUFactorNumeric");
  if (!mat->assembled) SETERRQ(1,"MatLUFactorNumeric:Not for unassembled matrix");
  if (mat->M != (*fact)->M || mat->N != (*fact)->N)
    SETERRQ(PETSC_ERR_SIZ,"MatLUFactorNumeric:Mat mat,Mat *fact: global dim");

  PLogEventBegin(MAT_LUFactorNumeric,mat,*fact,0,0); 
  ierr = (*mat->ops.lufactornumeric)(mat,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_LUFactorNumeric,mat,*fact,0,0); 
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer  viewer;
    ierr = ViewerDrawOpenX((*fact)->comm,0,0,0,0,300,300,&viewer);CHKERRQ(ierr);
    ierr = MatView(*fact,viewer); CHKERRQ(ierr);
    ierr = ViewerFlush(viewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
  }
  return 0;
}
/*@  
   MatCholeskyFactor - Performs in-place Cholesky factorization of a
   symmetric matrix. 

   Input Parameters:
.  mat - the matrix
.  perm - row and column permutations
.  f - expected fill as ratio of original fill

   Notes:
   See MatLUFactor() for the nonsymmetric case.  See also
   MatCholeskyFactorSymbolic(), and MatCholeskyFactorNumeric().

.keywords: matrix, factor, in-place, Cholesky

.seealso: MatLUFactor(), MatCholeskyFactorSymbolic(), MatCholeskyFactorNumeric()
@*/
int MatCholeskyFactor(Mat mat,IS perm,double f)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.choleskyfactor) SETERRQ(PETSC_ERR_SUP,"MatCholeskyFactor");
  if (!mat->assembled) SETERRQ(1,"MatCholeskyFactor:Not for unassembled matrix");

  PLogEventBegin(MAT_CholeskyFactor,mat,perm,0,0); 
  ierr = (*mat->ops.choleskyfactor)(mat,perm,f); CHKERRQ(ierr);
  PLogEventEnd(MAT_CholeskyFactor,mat,perm,0,0); 
  return 0;
}
/*@  
   MatCholeskyFactorSymbolic - Performs symbolic Cholesky factorization
   of a symmetric matrix. 

   Input Parameters:
.  mat - the matrix
.  perm - row and column permutations
.  f - expected fill as ratio of original

   Output Parameter:
.  fact - the factored matrix

   Notes:
   See MatLUFactorSymbolic() for the nonsymmetric case.  See also
   MatCholeskyFactor() and MatCholeskyFactorNumeric().

.keywords: matrix, factor, factorization, symbolic, Cholesky

.seealso: MatLUFactorSymbolic(), MatCholeskyFactor(), MatCholeskyFactorNumeric()
@*/
int MatCholeskyFactorSymbolic(Mat mat,IS perm,double f,Mat *fact)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!fact) SETERRQ(1,"MatCholeskyFactorSymbolic:Missing factor matrix argument");
  if (!mat->ops.choleskyfactorsymbolic)SETERRQ(PETSC_ERR_SUP,"MatCholeskyFactorSymbolic");
  if (!mat->assembled) SETERRQ(1,"MatCholeskyFactorSymbolic:Not for unassembled matrix");

  PLogEventBegin(MAT_CholeskyFactorSymbolic,mat,perm,0,0);
  ierr = (*mat->ops.choleskyfactorsymbolic)(mat,perm,f,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_CholeskyFactorSymbolic,mat,perm,0,0);
  return 0;
}
/*@  
   MatCholeskyFactorNumeric - Performs numeric Cholesky factorization
   of a symmetric matrix. Call this routine after first calling
   MatCholeskyFactorSymbolic().

   Input Parameter:
.  mat - the initial matrix

   Output Parameter:
.  fact - the factored matrix

.keywords: matrix, factor, numeric, Cholesky

.seealso: MatCholeskyFactorSymbolic(), MatCholeskyFactor(), MatLUFactorNumeric()
@*/
int MatCholeskyFactorNumeric(Mat mat,Mat *fact)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!fact) SETERRQ(1,"MatCholeskyFactorNumeric:Missing factor matrix argument");
  if (!mat->ops.choleskyfactornumeric) SETERRQ(PETSC_ERR_SUP,"MatCholeskyFactorNumeric");
  if (!mat->assembled) SETERRQ(1,"MatCholeskyFactorNumeric:Not for unassembled matrix");
  if (mat->M != (*fact)->M || mat->N != (*fact)->N)
    SETERRQ(PETSC_ERR_SIZ,"MatCholeskyFactorNumeric:Mat mat,Mat *fact: global dim");

  PLogEventBegin(MAT_CholeskyFactorNumeric,mat,*fact,0,0);
  ierr = (*mat->ops.choleskyfactornumeric)(mat,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_CholeskyFactorNumeric,mat,*fact,0,0);
  return 0;
}
/* ----------------------------------------------------------------*/
/*@
   MatSolve - Solves A x = b, given a factored matrix.

   Input Parameters:
.  mat - the factored matrix
.  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

.keywords: matrix, linear system, solve, LU, Cholesky, triangular solve

.seealso: MatSolveAdd(), MatSolveTrans(), MatSolveTransAdd()
@*/
int MatSolve(Mat mat,Vec b,Vec x)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE); PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(1,"MatSolve:x and y must be different vectors");
  if (!mat->factor) SETERRQ(1,"MatSolve:Unfactored matrix");
  if (mat->N != x->N) SETERRQ(PETSC_ERR_SIZ,"MatSolve:Mat mat,Vec x: global dim");
  if (mat->M != b->N) SETERRQ(PETSC_ERR_SIZ,"MatSolve:Mat mat,Vec b: global dim");
  if (mat->m != b->n) SETERRQ(PETSC_ERR_SIZ,"MatSolve:Mat mat,Vec b: local dim"); 

  if (!mat->ops.solve) SETERRQ(PETSC_ERR_SUP,"MatSolve");
  PLogEventBegin(MAT_Solve,mat,b,x,0); 
  ierr = (*mat->ops.solve)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_Solve,mat,b,x,0); 
  return 0;
}

/* @
   MatForwardSolve - Solves L x = b, given a factored matrix, A = LU.

   Input Parameters:
.  mat - the factored matrix
.  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   MatSolve() should be used for most applications, as it performs
   a forward solve followed by a backward solve.

.keywords: matrix, forward, LU, Cholesky, triangular solve

.seealso: MatSolve(), MatBackwardSolve()
@ */
int MatForwardSolve(Mat mat,Vec b,Vec x)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(1,"MatForwardSolve:x and y must be different vectors");
  if (!mat->factor) SETERRQ(1,"MatForwardSolve:Unfactored matrix");
  if (!mat->ops.forwardsolve) SETERRQ(PETSC_ERR_SUP,"MatForwardSolve");
  if (mat->N != x->N) SETERRQ(PETSC_ERR_SIZ,"MatForwardSolve:Mat mat,Vec x: global dim");
  if (mat->M != b->N) SETERRQ(PETSC_ERR_SIZ,"MatForwardSolve:Mat mat,Vec b: global dim");
  if (mat->m != b->n) SETERRQ(PETSC_ERR_SIZ,"MatForwardSolve:Mat mat,Vec b: local dim"); 

  PLogEventBegin(MAT_ForwardSolve,mat,b,x,0); 
  ierr = (*mat->ops.forwardsolve)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_ForwardSolve,mat,b,x,0); 
  return 0;
}

/* @
   MatBackwardSolve - Solves U x = b, given a factored matrix, A = LU.

   Input Parameters:
.  mat - the factored matrix
.  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

   Notes:
   MatSolve() should be used for most applications, as it performs
   a forward solve followed by a backward solve.

.keywords: matrix, backward, LU, Cholesky, triangular solve

.seealso: MatSolve(), MatForwardSolve()
@ */
int MatBackwardSolve(Mat mat,Vec b,Vec x)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(1,"MatBackwardSolve:x and b must be different vectors");
  if (!mat->factor) SETERRQ(1,"MatBackwardSolve:Unfactored matrix");
  if (!mat->ops.backwardsolve) SETERRQ(PETSC_ERR_SUP,"MatBackwardSolve");
  if (mat->N != x->N) SETERRQ(PETSC_ERR_SIZ,"MatBackwardSolve:Mat mat,Vec x: global dim");
  if (mat->M != b->N) SETERRQ(PETSC_ERR_SIZ,"MatBackwardSolve:Mat mat,Vec b: global dim");
  if (mat->m != b->n) SETERRQ(PETSC_ERR_SIZ,"MatBackwardSolve:Mat mat,Vec b: local dim"); 

  PLogEventBegin(MAT_BackwardSolve,mat,b,x,0); 
  ierr = (*mat->ops.backwardsolve)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_BackwardSolve,mat,b,x,0); 
  return 0;
}

/*@
   MatSolveAdd - Computes x = y + inv(A)*b, given a factored matrix.

   Input Parameters:
.  mat - the factored matrix
.  b - the right-hand-side vector
.  y - the vector to be added to 

   Output Parameter:
.  x - the result vector

.keywords: matrix, linear system, solve, LU, Cholesky, add

.seealso: MatSolve(), MatSolveTrans(), MatSolveTransAdd()
@*/
int MatSolveAdd(Mat mat,Vec b,Vec y,Vec x)
{
  Scalar one = 1.0;
  Vec    tmp;
  int    ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(1,"MatSolveAdd:x and b must be different vectors");
  if (!mat->factor) SETERRQ(1,"MatSolveAdd:Unfactored matrix");
  if (mat->N != x->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveAdd:Mat mat,Vec x: global dim");
  if (mat->M != b->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveAdd:Mat mat,Vec b: global dim");
  if (mat->M != y->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveAdd:Mat mat,Vec y: global dim");
  if (mat->m != b->n) SETERRQ(PETSC_ERR_SIZ,"MatSolveAdd:Mat mat,Vec b: local dim"); 
  if (x->n != y->n) SETERRQ(PETSC_ERR_SIZ,"MatSolveAdd:Vec x,Vec y: local dim"); 

  PLogEventBegin(MAT_SolveAdd,mat,b,x,y); 
  if (mat->ops.solveadd)  {
    ierr = (*mat->ops.solveadd)(mat,b,y,x); CHKERRQ(ierr);
  } 
  else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolve(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,y,x); CHKERRQ(ierr);
    }
    else {
      ierr = VecDuplicate(x,&tmp); CHKERRQ(ierr);
      PLogObjectParent(mat,tmp);
      ierr = VecCopy(x,tmp); CHKERRQ(ierr);
      ierr = MatSolve(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,tmp,x); CHKERRQ(ierr);
      ierr = VecDestroy(tmp); CHKERRQ(ierr);
    }
  }
  PLogEventEnd(MAT_SolveAdd,mat,b,x,y); 
  return 0;
}
/*@
   MatSolveTrans - Solves A' x = b, given a factored matrix.

   Input Parameters:
.  mat - the factored matrix
.  b - the right-hand-side vector

   Output Parameter:
.  x - the result vector

.keywords: matrix, linear system, solve, LU, Cholesky, transpose

.seealso: MatSolve(), MatSolveAdd(), MatSolveTransAdd()
@*/
int MatSolveTrans(Mat mat,Vec b,Vec x)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (!mat->factor) SETERRQ(1,"MatSolveTrans:Unfactored matrix");
  if (x == b) SETERRQ(1,"MatSolveTrans:x and b must be different vectors");
  if (!mat->ops.solvetrans) SETERRQ(PETSC_ERR_SUP,"MatSolveTrans");
  if (mat->M != x->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveTrans:Mat mat,Vec x: global dim");
  if (mat->N != b->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveTrans:Mat mat,Vec b: global dim");

  PLogEventBegin(MAT_SolveTrans,mat,b,x,0); 
  ierr = (*mat->ops.solvetrans)(mat,b,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_SolveTrans,mat,b,x,0); 
  return 0;
}
/*@
   MatSolveTransAdd - Computes x = y + inv(trans(A)) b, given a 
                      factored matrix. 

   Input Parameters:
.  mat - the factored matrix
.  b - the right-hand-side vector
.  y - the vector to be added to 

   Output Parameter:
.  x - the result vector

.keywords: matrix, linear system, solve, LU, Cholesky, transpose, add  

.seealso: MatSolve(), MatSolveAdd(), MatSolveTrans()
@*/
int MatSolveTransAdd(Mat mat,Vec b,Vec y,Vec x)
{
  Scalar one = 1.0;
  int    ierr;
  Vec    tmp;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (x == b) SETERRQ(1,"MatSolveTransAdd:x and b must be different vectors");
  if (!mat->factor) SETERRQ(1,"MatSolveTransAdd:Unfactored matrix");
  if (mat->M != x->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveTransAdd:Mat mat,Vec x: global dim");
  if (mat->N != b->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveTransAdd:Mat mat,Vec b: global dim");
  if (mat->N != y->N) SETERRQ(PETSC_ERR_SIZ,"MatSolveTransAdd:Mat mat,Vec y: global dim");
  if (x->n != y->n) SETERRQ(PETSC_ERR_SIZ,"MatSolveTransAdd:Vec x,Vec y: local dim");

  PLogEventBegin(MAT_SolveTransAdd,mat,b,x,y); 
  if (mat->ops.solvetransadd) {
    ierr = (*mat->ops.solvetransadd)(mat,b,y,x); CHKERRQ(ierr);
  }
  else {
    /* do the solve then the add manually */
    if (x != y) {
      ierr = MatSolveTrans(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,y,x); CHKERRQ(ierr);
    }
    else {
      ierr = VecDuplicate(x,&tmp); CHKERRQ(ierr);
      PLogObjectParent(mat,tmp);
      ierr = VecCopy(x,tmp); CHKERRQ(ierr);
      ierr = MatSolveTrans(mat,b,x); CHKERRQ(ierr);
      ierr = VecAXPY(&one,tmp,x); CHKERRQ(ierr);
      ierr = VecDestroy(tmp); CHKERRQ(ierr);
    }
  }
  PLogEventEnd(MAT_SolveTransAdd,mat,b,x,y); 
  return 0;
}
/* ----------------------------------------------------------------*/

/*@
   MatRelax - Computes one relaxation sweep.

   Input Parameters:
.  mat - the matrix
.  b - the right hand side
.  omega - the relaxation factor
.  flag - flag indicating the type of SOR, one of
$     SOR_FORWARD_SWEEP
$     SOR_BACKWARD_SWEEP
$     SOR_SYMMETRIC_SWEEP (SSOR method)
$     SOR_LOCAL_FORWARD_SWEEP
$     SOR_LOCAL_BACKWARD_SWEEP
$     SOR_LOCAL_SYMMETRIC_SWEEP (local SSOR)
$     SOR_APPLY_UPPER, SOR_APPLY_LOWER - applies 
$       upper/lower triangular part of matrix to
$       vector (with omega)
$     SOR_ZERO_INITIAL_GUESS - zero initial guess
.  shift -  diagonal shift
.  its - the number of iterations

   Output Parameters:
.  x - the solution (can contain an initial guess)

   Notes:
   SOR_LOCAL_FORWARD_SWEEP, SOR_LOCAL_BACKWARD_SWEEP, and
   SOR_LOCAL_SYMMETRIC_SWEEP perform seperate independent smoothings
   on each processor. 

   Application programmers will not generally use MatRelax() directly,
   but instead will employ the SLES/PC interface.

   Notes for Advanced Users:
   The flags are implemented as bitwise inclusive or operations.
   For example, use (SOR_ZERO_INITIAL_GUESS | SOR_SYMMETRIC_SWEEP)
   to specify a zero initial guess for SSOR.

.keywords: matrix, relax, relaxation, sweep
@*/
int MatRelax(Mat mat,Vec b,double omega,MatSORType flag,double shift,
             int its,Vec x)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(b,VEC_COOKIE);  PetscValidHeaderSpecific(x,VEC_COOKIE);
  if (!mat->ops.relax) SETERRQ(PETSC_ERR_SUP,"MatRelax");
  if (!mat->assembled) SETERRQ(1,"MatRelax:Not for unassembled matrix");
  if (mat->N != x->N) SETERRQ(PETSC_ERR_SIZ,"MatRelax:Mat mat,Vec x: global dim");
  if (mat->M != b->N) SETERRQ(PETSC_ERR_SIZ,"MatRelax:Mat mat,Vec b: global dim");
  if (mat->m != b->n) SETERRQ(PETSC_ERR_SIZ,"MatRelax:Mat mat,Vec b: local dim");

  PLogEventBegin(MAT_Relax,mat,b,x,0); 
  ierr =(*mat->ops.relax)(mat,b,omega,flag,shift,its,x); CHKERRQ(ierr);
  PLogEventEnd(MAT_Relax,mat,b,x,0); 
  return 0;
}

/*
      Default matrix copy routine.
*/
int MatCopy_Basic(Mat A,Mat B)
{
  int    ierr,i,rstart,rend,nz,*cwork;
  Scalar *vwork;

  ierr = MatZeroEntries(B); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&i,nz,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

/*@C  
   MatCopy - Copys a matrix to another matrix.

   Input Parameters:
.  A - the matrix

   Output Parameter:
.  B - where the copy is put

   Notes:
   MatCopy() copies the matrix entries of a matrix to another existing
   matrix (after first zeroing the second matrix).  A related routine is
   MatConvert(), which first creates a new matrix and then copies the data.
   
.keywords: matrix, copy, convert

.seealso: MatConvert()
@*/
int MatCopy(Mat A,Mat B)
{
  int ierr;
  PetscValidHeaderSpecific(A,MAT_COOKIE); PetscValidHeaderSpecific(B,MAT_COOKIE);
  if (!A->assembled) SETERRQ(1,"MatCopy:Not for unassembled matrix");
  if (A->M != B->M || A->N != B->N) SETERRQ(PETSC_ERR_SIZ,"MatCopy:Mat A,Mat B: global dim");

  PLogEventBegin(MAT_Copy,A,B,0,0); 
  if (A->ops.copy) { 
    ierr = (*A->ops.copy)(A,B); CHKERRQ(ierr);
  }
  else { /* generic conversion */
    ierr = MatCopy_Basic(A,B); CHKERRQ(ierr);
  }
  PLogEventEnd(MAT_Copy,A,B,0,0); 
  return 0;
}

/*@C  
   MatConvert - Converts a matrix to another matrix, either of the same
   or different type.

   Input Parameters:
.  mat - the matrix
.  newtype - new matrix type.  Use MATSAME to create a new matrix of the
   same type as the original matrix.

   Output Parameter:
.  M - pointer to place new matrix

   Notes:
   MatConvert() first creates a new matrix and then copies the data from
   the first matrix.  A related routine is MatCopy(), which copies the matrix
   entries of one matrix to another already existing matrix context.

.keywords: matrix, copy, convert

.seealso: MatCopy()
@*/
int MatConvert(Mat mat,MatType newtype,Mat *M)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!M) SETERRQ(1,"MatConvert:Bad new matrix address");
  if (!mat->assembled) SETERRQ(1,"MatConvert:Not for unassembled matrix");

  PLogEventBegin(MAT_Convert,mat,0,0,0); 
  if (newtype == mat->type || newtype == MATSAME) {
    if (mat->ops.convertsametype) { /* customized copy */
      ierr = (*mat->ops.convertsametype)(mat,M,COPY_VALUES); CHKERRQ(ierr);
    }
    else { /* generic conversion */
      ierr = MatConvert_Basic(mat,newtype,M); CHKERRQ(ierr);
    }
  }
  else if (mat->ops.convert) { /* customized conversion */
    ierr = (*mat->ops.convert)(mat,newtype,M); CHKERRQ(ierr);
  }
  else { /* generic conversion */
    ierr = MatConvert_Basic(mat,newtype,M); CHKERRQ(ierr);
  }
  PLogEventEnd(MAT_Convert,mat,0,0,0); 
  return 0;
}

/*@ 
   MatGetDiagonal - Gets the diagonal of a matrix.

   Input Parameters:
.  mat - the matrix
.  v - the vector for storing the diagonal

   Output Parameter:
.  v - the diagonal of the matrix

.keywords: matrix, get, diagonal
@*/
int MatGetDiagonal(Mat mat,Vec v)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);PetscValidHeaderSpecific(v,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatGetDiagonal:Not for unassembled matrix");
  if (mat->ops.getdiagonal) return (*mat->ops.getdiagonal)(mat,v);
  SETERRQ(PETSC_ERR_SUP,"MatGetDiagonal");
}

/*@C
   MatTranspose - Computes an in-place or out-of-place transpose of a matrix.

   Input Parameter:
.  mat - the matrix to transpose

   Output Parameters:
.  B - the transpose (or pass in PETSC_NULL for an in-place transpose)

.keywords: matrix, transpose

.seealso: MatMultTrans(), MatMultTransAdd()
@*/
int MatTranspose(Mat mat,Mat *B)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatTranspose:Not for unassembled matrix");
  if (mat->ops.transpose) return (*mat->ops.transpose)(mat,B);
  SETERRQ(PETSC_ERR_SUP,"MatTranspose");
}

/*@
   MatEqual - Compares two matrices.

   Input Parameters:
.  A - the first matrix
.  B - the second matrix

   Output Parameter:
.  flg : PETSC_TRUE if the matrices are equal;
         PETSC_FALSE otherwise.

.keywords: matrix, equal, equivalent
@*/
int MatEqual(Mat A,Mat B,PetscTruth *flg)
{
  PetscValidHeaderSpecific(A,MAT_COOKIE); PetscValidHeaderSpecific(B,MAT_COOKIE);
  PetscValidIntPointer(flg);
  if (!A->assembled) SETERRQ(1,"MatEqual:Not for unassembled matrix");
  if (!B->assembled) SETERRQ(1,"MatEqual:Not for unassembled matrix");
  if (A->M != B->M || A->N != B->N) SETERRQ(PETSC_ERR_SIZ,"MatCopy:Mat A,Mat B: global dim");
  if (A->ops.equal) return (*A->ops.equal)(A,B,flg);
  SETERRQ(PETSC_ERR_SUP,"MatEqual");
}

/*@
   MatDiagonalScale - Scales a matrix on the left and right by diagonal
   matrices that are stored as vectors.  Either of the two scaling
   matrices can be PETSC_NULL.

   Input Parameters:
.  mat - the matrix to be scaled
.  l - the left scaling vector (or PETSC_NULL)
.  r - the right scaling vector (or PETSC_NULL)

   Notes:
   MatDiagonalScale() computes A <- LAR, where
$      L = a diagonal matrix
$      R = a diagonal matrix

.keywords: matrix, diagonal, scale

.seealso: MatDiagonalScale()
@*/
int MatDiagonalScale(Mat mat,Vec l,Vec r)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.diagonalscale) SETERRQ(PETSC_ERR_SUP,"MatDiagonalScale");
  if (l) PetscValidHeaderSpecific(l,VEC_COOKIE); 
  if (r) PetscValidHeaderSpecific(r,VEC_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatDiagonalScale:Not for unassembled matrix");

  PLogEventBegin(MAT_Scale,mat,0,0,0);
  ierr = (*mat->ops.diagonalscale)(mat,l,r); CHKERRQ(ierr);
  PLogEventEnd(MAT_Scale,mat,0,0,0);
  return 0;
} 

/*@
    MatScale - Scales all elements of a matrix by a given number.

    Input Parameters:
.   mat - the matrix to be scaled
.   a  - the scaling value

    Output Parameter:
.   mat - the scaled matrix

.keywords: matrix, scale

.seealso: MatDiagonalScale()
@*/
int MatScale(Scalar *a,Mat mat)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidScalarPointer(a);
  if (!mat->ops.scale) SETERRQ(PETSC_ERR_SUP,"MatScale");
  if (!mat->assembled) SETERRQ(1,"MatScale:Not for unassembled matrix");

  PLogEventBegin(MAT_Scale,mat,0,0,0);
  ierr = (*mat->ops.scale)(a,mat); CHKERRQ(ierr);
  PLogEventEnd(MAT_Scale,mat,0,0,0);
  return 0;
} 

/*@ 
   MatNorm - Calculates various norms of a matrix.

   Input Parameters:
.  mat - the matrix
.  type - the type of norm, NORM_1, NORM_2, NORM_FROBENIUS, NORM_INFINITY

   Output Parameters:
.  norm - the resulting norm 

.keywords: matrix, norm, Frobenius
@*/
int MatNorm(Mat mat,NormType type,double *norm)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidScalarPointer(norm);

  if (!norm) SETERRQ(1,"MatNorm:bad addess for value");
  if (!mat->assembled) SETERRQ(1,"MatNorm:Not for unassembled matrix");
  if (mat->ops.norm) return (*mat->ops.norm)(mat,type,norm);
  SETERRQ(PETSC_ERR_SUP,"MatNorm:Not for this matrix type");
}

/*@
   MatAssemblyBegin - Begins assembling the matrix.  This routine should
   be called after completing all calls to MatSetValues().

   Input Parameters:
.  mat - the matrix 
.  type - type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY
 
   Notes: 
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

.keywords: matrix, assembly, assemble, begin

.seealso: MatAssemblyEnd(), MatSetValues()
@*/
int MatAssemblyBegin(Mat mat,MatAssemblyType type)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->assembled) {
    mat->was_assembled = PETSC_TRUE; 
    mat->assembled     = PETSC_FALSE;
  }
  PLogEventBegin(MAT_AssemblyBegin,mat,0,0,0);
  if (mat->ops.assemblybegin){ierr = (*mat->ops.assemblybegin)(mat,type);CHKERRQ(ierr);}
  PLogEventEnd(MAT_AssemblyBegin,mat,0,0,0);
  return 0;
}

/*@
   MatAssemblyEnd - Completes assembling the matrix.  This routine should
   be called after MatAssemblyBegin().

   Input Parameters:
.  mat - the matrix 
.  type - type of assembly, either MAT_FLUSH_ASSEMBLY or MAT_FINAL_ASSEMBLY

   Options Database Keys:
$  -mat_view_info : Prints info on matrix at
$      conclusion of MatEndAssembly()
$  -mat_view_info_detailed: Prints more detailed info.
$  -mat_view : Prints matrix in ASCII format.
$  -mat_view_matlab : Prints matrix in Matlab format.
$  -mat_view_draw : Draws nonzero structure of matrix,
$      using MatView() and DrawOpenX().
$  -display <name> : Set display name (default is host)
$  -draw_pause <sec> : Set number of seconds to pause after display

   Notes: 
   MatSetValues() generally caches the values.  The matrix is ready to
   use only after MatAssemblyBegin() and MatAssemblyEnd() have been called.
   Use MAT_FLUSH_ASSEMBLY when switching between ADD_VALUES and INSERT_VALUES
   in MatSetValues(); use MAT_FINAL_ASSEMBLY for the final assembly before
   using the matrix.

.keywords: matrix, assembly, assemble, end

.seealso: MatAssemblyBegin(), MatSetValues(), DrawOpenX(), MatView()
@*/
int MatAssemblyEnd(Mat mat,MatAssemblyType type)
{
  int        ierr,flg;
  static int inassm = 0;

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  inassm++;
  PLogEventBegin(MAT_AssemblyEnd,mat,0,0,0);
  if (mat->ops.assemblyend) {
    ierr = (*mat->ops.assemblyend)(mat,type); CHKERRQ(ierr);
  }
  mat->assembled = PETSC_TRUE; mat->num_ass++;
  PLogEventEnd(MAT_AssemblyEnd,mat,0,0,0);

  if (inassm == 1) {
    ierr = OptionsHasName(PETSC_NULL,"-mat_view_info",&flg); CHKERRQ(ierr);
    if (flg) {
      Viewer viewer;
      ierr = ViewerFileOpenASCII(mat->comm,"stdout",&viewer);CHKERRQ(ierr);
      ierr = ViewerSetFormat(viewer,ASCII_FORMAT_INFO,0);CHKERRQ(ierr);
      ierr = MatView(mat,viewer); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    ierr = OptionsHasName(PETSC_NULL,"-mat_view_info_detailed",&flg);CHKERRQ(ierr);
    if (flg) {
      Viewer viewer;
      ierr = ViewerFileOpenASCII(mat->comm,"stdout",&viewer);CHKERRQ(ierr);
      ierr = ViewerSetFormat(viewer,ASCII_FORMAT_INFO_DETAILED,0);CHKERRQ(ierr);
      ierr = MatView(mat,viewer); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    ierr = OptionsHasName(PETSC_NULL,"-mat_view",&flg); CHKERRQ(ierr);
    if (flg) {
      Viewer viewer;
      ierr = ViewerFileOpenASCII(mat->comm,"stdout",&viewer);CHKERRQ(ierr);
      ierr = MatView(mat,viewer); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    ierr = OptionsHasName(PETSC_NULL,"-mat_view_matlab",&flg); CHKERRQ(ierr);
    if (flg) {
      Viewer viewer;
      ierr = ViewerFileOpenASCII(mat->comm,"stdout",&viewer);CHKERRQ(ierr);
      ierr = ViewerSetFormat(viewer,ASCII_FORMAT_MATLAB,"M");CHKERRQ(ierr);
      ierr = MatView(mat,viewer); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
    ierr = OptionsHasName(PETSC_NULL,"-mat_view_draw",&flg); CHKERRQ(ierr);
    if (flg) {
      Viewer    viewer;
      ierr = ViewerDrawOpenX(mat->comm,0,0,0,0,300,300,&viewer); CHKERRQ(ierr);
      ierr = MatView(mat,viewer); CHKERRQ(ierr);
      ierr = ViewerFlush(viewer); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
    }
  }
  inassm--;
  return 0;
}

/*@
   MatCompress - Tries to store the matrix in as little space as 
   possible.  May fail if memory is already fully used, since it
   tries to allocate new space.

   Input Parameters:
.  mat - the matrix 

.keywords: matrix, compress
@*/
int MatCompress(Mat mat)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->ops.compress) return (*mat->ops.compress)(mat);
  return 0;
}
/*@
   MatSetOption - Sets a parameter option for a matrix. Some options
   may be specific to certain storage formats.  Some options
   determine how values will be inserted (or added). Sorted, 
   row-oriented input will generally assemble the fastest. The default
   is row-oriented, nonsorted input. 

   Input Parameters:
.  mat - the matrix 
.  option - the option, one of the following:
$    MAT_ROW_ORIENTED
$    MAT_COLUMN_ORIENTED,
$    MAT_ROWS_SORTED,
$    MAT_COLUMNS_SORTED,
$    MAT_NO_NEW_NONZERO_LOCATIONS, 
$    MAT_YES_NEW_NONZERO_LOCATIONS, 
$    MAT_SYMMETRIC,
$    MAT_STRUCTURALLY_SYMMETRIC,
$    MAT_NO_NEW_DIAGONALS,
$    MAT_YES_NEW_DIAGONALS,
$    and possibly others.  

   Notes:
   Some options are relevant only for particular matrix types and
   are thus ignored by others.  Other options are not supported by
   certain matrix types and will generate an error message if set.

   If using a Fortran 77 module to compute a matrix, one may need to 
   use the column-oriented option (or convert to the row-oriented 
   format).  

   MAT_NO_NEW_NONZERO_LOCATIONS indicates that any add or insertion 
   that will generate a new entry in the nonzero structure is ignored.
   What this means is if memory is not allocated for this particular 
   lot, then the insertion is ignored. For dense matrices, where  
   the entire array is allocated, no entries are ever ignored. 

.keywords: matrix, option, row-oriented, column-oriented, sorted, nonzero
@*/
int MatSetOption(Mat mat,MatOption op)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (mat->ops.setoption) return (*mat->ops.setoption)(mat,op);
  return 0;
}

/*@
   MatZeroEntries - Zeros all entries of a matrix.  For sparse matrices
   this routine retains the old nonzero structure.

   Input Parameters:
.  mat - the matrix 

.keywords: matrix, zero, entries

.seealso: MatZeroRows()
@*/
int MatZeroEntries(Mat mat)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.zeroentries) SETERRQ(PETSC_ERR_SUP,"MatZeroEntries");

  PLogEventBegin(MAT_ZeroEntries,mat,0,0,0);
  ierr = (*mat->ops.zeroentries)(mat); CHKERRQ(ierr);
  PLogEventEnd(MAT_ZeroEntries,mat,0,0,0);
  return 0;
}

/*@ 
   MatZeroRows - Zeros all entries (except possibly the main diagonal)
   of a set of rows of a matrix.

   Input Parameters:
.  mat - the matrix
.  is - index set of rows to remove
.  diag - pointer to value put in all diagonals of eliminated rows.
          Note that diag is not a pointer to an array, but merely a
          pointer to a single value.

   Notes:
   For the AIJ matrix formats this removes the old nonzero structure,
   but does not release memory.  For the dense and block diagonal
   formats this does not alter the nonzero structure.

   The user can set a value in the diagonal entry (or for the AIJ and
   row formats can optionally remove the main diagonal entry from the
   nonzero structure as well, by passing a null pointer as the final
   argument).

.keywords: matrix, zero, rows, boundary conditions 

.seealso: MatZeroEntries(), 
@*/
int MatZeroRows(Mat mat,IS is, Scalar *diag)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatZeroRows:Not for unassembled matrix");
  if (mat->ops.zerorows) return (*mat->ops.zerorows)(mat,is,diag);
  SETERRQ(PETSC_ERR_SUP,"MatZeroRows");
}

/*@
   MatGetSize - Returns the numbers of rows and columns in a matrix.

   Input Parameter:
.  mat - the matrix

   Output Parameters:
.  m - the number of global rows
.  n - the number of global columns

.keywords: matrix, dimension, size, rows, columns, global, get

.seealso: MatGetLocalSize()
@*/
int MatGetSize(Mat mat,int *m,int* n)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(m);
  PetscValidIntPointer(n);
  return (*mat->ops.getsize)(mat,m,n);
}

/*@
   MatGetLocalSize - Returns the number of rows and columns in a matrix
   stored locally.  This information may be implementation dependent, so
   use with care.

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  m - the number of local rows
.  n - the number of local columns

.keywords: matrix, dimension, size, local, rows, columns, get

.seealso: MatGetSize()
@*/
int MatGetLocalSize(Mat mat,int *m,int* n)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(m);
  PetscValidIntPointer(n);
  return (*mat->ops.getlocalsize)(mat,m,n);
}

/*@
   MatGetOwnershipRange - Returns the range of matrix rows owned by
   this processor, assuming that the matrix is laid out with the first
   n1 rows on the first processor, the next n2 rows on the second, etc.
   For certain parallel layouts this range may not be well-defined.

   Input Parameters:
.  mat - the matrix

   Output Parameters:
.  m - the first local row
.  n - one more then the last local row

.keywords: matrix, get, range, ownership
@*/
int MatGetOwnershipRange(Mat mat,int *m,int* n)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(m);
  PetscValidIntPointer(n);
  if (mat->ops.getownershiprange) return (*mat->ops.getownershiprange)(mat,m,n);
  SETERRQ(PETSC_ERR_SUP,"MatGetOwnershipRange");
}

/*@  
   MatILUFactorSymbolic - Performs symbolic ILU factorization of a matrix.
   Uses levels of fill only, not drop tolerance. Use MatLUFactorNumeric() 
   to complete the factorization.

   Input Parameters:
.  mat - the matrix
.  row - row permutation
.  column - column permutation
.  fill - number of levels of fill
.  f - expected fill as ratio of the original number of nonzeros, 
       for example 3.0; choosing this parameter well can result in 
       more efficient use of time and space.

   Output Parameters:
.  fact - new matrix that has been symbolically factored

   Options Database Key:
$   -mat_ilu_fill <f>, where f is the fill ratio

   Notes:
   See the file $(PETSC_DIR)/Performace for additional information about
   choosing the fill factor for better efficiency.

.keywords: matrix, factor, incomplete, ILU, symbolic, fill

.seealso: MatLUFactorSymbolic(), MatLUFactorNumeric()
@*/
int MatILUFactorSymbolic(Mat mat,IS row,IS col,double f,int fill,Mat *fact)
{
  int ierr,flg;

  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (fill < 0) SETERRQ(1,"MatILUFactorSymbolic:Levels of fill negative");
  if (!fact) SETERRQ(1,"MatILUFactorSymbolic:Fact argument is missing");
  if (!mat->ops.ilufactorsymbolic) SETERRQ(PETSC_ERR_SUP,"MatILUFactorSymbolic");
  if (!mat->assembled) SETERRQ(1,"MatILUFactorSymbolic:Not for unassembled matrix");

  ierr = OptionsGetDouble(PETSC_NULL,"-mat_ilu_fill",&f,&flg); CHKERRQ(ierr);
  PLogEventBegin(MAT_ILUFactorSymbolic,mat,row,col,0);
  ierr = (*mat->ops.ilufactorsymbolic)(mat,row,col,f,fill,fact); CHKERRQ(ierr);
  PLogEventEnd(MAT_ILUFactorSymbolic,mat,row,col,0);
  return 0;
}

/*@  
   MatIncompleteCholeskyFactorSymbolic - Performs symbolic incomplete
   Cholesky factorization for a symmetric matrix.  Use 
   MatCholeskyFactorNumeric() to complete the factorization.

   Input Parameters:
.  mat - the matrix
.  perm - row and column permutation
.  fill - levels of fill
.  f - expected fill as ratio of original fill

   Output Parameter:
.  fact - the factored matrix

   Note:  Currently only no-fill factorization is supported.

.keywords: matrix, factor, incomplete, ICC, Cholesky, symbolic, fill

.seealso: MatCholeskyFactorNumeric(), MatCholeskyFactor()
@*/
int MatIncompleteCholeskyFactorSymbolic(Mat mat,IS perm,double f,int fill,
                                        Mat *fact)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (fill < 0) SETERRQ(1,"MatIncompleteCholeskyFactorSymbolic:Fill negative");
  if (!fact) SETERRQ(1,"MatIncompleteCholeskyFactorSymbolic:Missing fact argument");
  if (!mat->ops.incompletecholeskyfactorsymbolic) 
     SETERRQ(PETSC_ERR_SUP,"MatIncompleteCholeskyFactorSymbolic");
  if (!mat->assembled)
     SETERRQ(1,"MatIncompleteCholeskyFactorSymbolic:Not for unassembled matrix");

  PLogEventBegin(MAT_IncompleteCholeskyFactorSymbolic,mat,perm,0,0);
  ierr = (*mat->ops.incompletecholeskyfactorsymbolic)(mat,perm,f,fill,fact);CHKERRQ(ierr);
  PLogEventEnd(MAT_IncompleteCholeskyFactorSymbolic,mat,perm,0,0);
  return 0;
}

/*@C
   MatGetArray - Returns a pointer to the element values in the matrix.
   This routine  is implementation dependent, and may not even work for 
   certain matrix types. You MUST call MatRestoreArray() when you no 
   longer need to access the array.

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  v - the location of the values

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the Fortran chapter of the users manual and 
   petsc/src/mat/examples for details.

.keywords: matrix, array, elements, values

.seeaols: MatRestoreArray()
@*/
int MatGetArray(Mat mat,Scalar **v)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!v) SETERRQ(1,"MatGetArray:Bad input, array pointer location");
  if (!mat->ops.getarray) SETERRQ(PETSC_ERR_SUP,"MatGetArray");
  return (*mat->ops.getarray)(mat,v);
}

/*@C
   MatRestoreArray - Restores the matrix after MatGetArray has been called.

   Input Parameter:
.  mat - the matrix
.  v - the location of the values

   Fortran Note:
   The Fortran interface is slightly different from that given below.
   See the users manual and petsc/src/mat/examples for details.

.keywords: matrix, array, elements, values, resrore

.seealso: MatGetArray()
@*/
int MatRestoreArray(Mat mat,Scalar **v)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!v) SETERRQ(1,"MatRestoreArray:Bad input, array pointer location");
  if (!mat->ops.restorearray) SETERRQ(PETSC_ERR_SUP,"MatResroreArray");
  return (*mat->ops.restorearray)(mat,v);
}

/*@C
   MatGetSubMatrices - Extracts several submatrices from a matrix. If submat
   points to an array of valid matrices, it may be reused.

   Input Parameters:
.  mat - the matrix
.  irow, icol - index sets of rows and columns to extract
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX

   Output Parameter:
.  submat - the submatrices


.keywords: matrix, get, submatrix, submatrices

.seealso: MatDestroyMatrices()
@*/
int MatGetSubMatrices(Mat mat,int n, IS *irow,IS *icol,MatGetSubMatrixCall scall,
                      Mat **submat)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->ops.getsubmatrices) SETERRQ(PETSC_ERR_SUP,"MatGetSubMatrices");
  if (!mat->assembled) SETERRQ(1,"MatGetSubMatrices:Not for unassembled matrix");

  PLogEventBegin(MAT_GetSubMatrices,mat,0,0,0);
  ierr = (*mat->ops.getsubmatrices)(mat,n,irow,icol,scall,submat); CHKERRQ(ierr);
  PLogEventEnd(MAT_GetSubMatrices,mat,0,0,0);
  return 0;
}

/*@C
   MatDestroyMatrices - Destroys a set of matrices obtained with MatGetSubMatrices()

   Input Parameters:
.  n - the number of local matrices
.  mat - the matrices

.keywords: matrix, get, submatrix, submatrices

.seealso: MatGetSubMatrices()
@*/
int MatDestroyMatrices(int n,  Mat **mat)
{
  int ierr,i;

  for ( i=0; i<n; i++ ) {
    ierr = MatDestroy((*mat)[i]); CHKERRQ(ierr);
  }
  return 0;
}

/*@
   MatIncreaseOverlap - Given a set of submatrices indicated by index sets,
   replaces the index by larger ones that represent submatrices with more
   overlap.

   Input Parameters:
.  mat - the matrix
.  n   - the number of index sets
.  is  - the array of pointers to index sets
.  ov  - the additional overlap requested

.keywords: matrix, overlap, Schwarz

.seealso: MatGetSubMatrices()
@*/
int MatIncreaseOverlap(Mat mat,int n, IS *is,int ov)
{
  int ierr;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(1,"MatIncreaseOverlap:Not for unassembled matrix");

  if (ov == 0) return 0;
  if (!mat->ops.increaseoverlap) SETERRQ(PETSC_ERR_SUP,"MatIncreaseOverlap");
  PLogEventBegin(MAT_IncreaseOverlap,mat,0,0,0);
  ierr = (*mat->ops.increaseoverlap)(mat,n,is,ov); CHKERRQ(ierr);
  PLogEventEnd(MAT_IncreaseOverlap,mat,0,0,0);
  return 0;
}

/*@
   MatPrintHelp - Prints all the options for the matrix.

   Input Parameter:
.  mat - the matrix 

   Options Database Keys:
$  -help, -h

.keywords: mat, help

.seealso: MatCreate(), MatCreateXXX()
@*/
int MatPrintHelp(Mat mat)
{
  static int called = 0;
  MPI_Comm   comm = mat->comm;

  if (!called) {
    PetscPrintf(comm,"General matrix options:\n");
    PetscPrintf(comm,"  -mat_view_info : view basic matrix info during MatAssemblyEnd()\n");
    PetscPrintf(comm,"  -mat_view_info_detailed : view detailed matrix info during MatAssemblyEnd()\n");
    PetscPrintf(comm,"  -mat_view_draw : draw nonzero matrix structure during MatAssemblyEnd()\n");
    PetscPrintf(comm,"      -draw_pause <sec> : set seconds of display pause\n");
    PetscPrintf(comm,"      -display <name> : set alternate display\n");
    called = 1;
  }
  if (mat->ops.printhelp) (*mat->ops.printhelp)(mat);
  return 0;
}

/*@
   MatGetBlockSize - Returns the block size. useful especially for
    MATBAIJ, MATBDIAG formats
   

   Input Parameter:
.  mat - the matrix

   Output Parameter:
.  bs - block size

.keywords: mat, block, size 

.seealso: MatCreateXXXBAIJ()
@*/
int MatGetBlockSize(Mat mat,int *bs)
{
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidIntPointer(bs);
  if (!mat->ops.getblocksize) SETERRQ(PETSC_ERR_SUP,"MatGetBlockSize");
  return (*mat->ops.getblocksize)(mat,bs);
}
