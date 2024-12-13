#include "monolis.h"
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I  "petscmat.h"  I*/
#include <iostream>

const char *const KSPMONOLISTypes[] = {KSPCG, KSPGROPPCG, KSPPIPECG, KSPPIPECR, KSPBCGS, KSPPIPEBCGS, "bicgstab-np", "cabicgstab-np", "pipebicg-np", "sor", "ir"};
const char *const KSPMONOLISPCTypes[] = {PCNONE, "diag", PCILU, PCJACOBI, PCSOR, "sainv", "rif", "spike", "direct", "mumps"};
PetscBool      isSeqAIJ, isMPIBAIJ, isMPIAIJ;
PetscInt       nloc;                              // number of DOFs owned by current cpu
PetscScalar    *matv;                             // matrix value, for tranlate it to monolis solver
PetscScalar    *rhs, *lhs;                        // rhs and lhs of equations
											
static PetscErrorCode KSPSetFromOptions_MONOLIS(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  MONOLIS  *data = (MONOLIS *)ksp->data;
  PetscInt    i;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("monolis_prm_initialize",monolis_prm_initialize(data));
  PetscStackCallExternalVoid("monolis_com_initialize",monolis_com_initialize(data));
  PetscStackCallExternalVoid("monolis_mat_initialize",monolis_mat_initialize(data));
  PetscOptionsHeadBegin(PetscOptionsObject, "KSPMONOLIS options, cf. https://github.com/nqomorita/monolis");
  i = monolis_iter_CG;
  PetscCall(PetscOptionsEList("-monolis_type", "Type of Krylov method", "KSPMONOLISGetType", KSPMONOLISTypes, PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISTypes), KSPMONOLISTypes[KSP_MONOLIS_TYPE_CG], &i, NULL));
  if (i == PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISTypes)-1) i = monolis_iter_CG-1;
  data->prm.method = i+1;
  i = monolis_prec_NONE; 
  PetscCall(PetscOptionsEList("-monolis_pc_type", "Type of PC method", "None", KSPMONOLISPCTypes, PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISPCTypes), KSPMONOLISPCTypes[MONOLIS_PC_NONE], &i, NULL));
  if (i == PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISPCTypes) - 1) i = monolis_prec_NONE;
  data->prm.precond = i;
  PetscCall(PetscOptionsInt("-ksp_max_it", "Max number of iterations", "None", 1000, &(data->prm.maxiter), NULL));
  PetscCall(PetscOptionsReal("-ksp_atol", "Max tolrence", "None", 1.e-8, &(data->prm.tol), NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-log_view",&flg));
  data->prm.show_summary = flg;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode KSPView_MONOLIS(KSP ksp, PetscViewer viewer)
{
  MONOLIS  *data = (MONOLIS *)ksp->data;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "MONOLIS type: %s\n", KSPMONOLISTypes[std::min(static_cast<PetscInt>(data->prm.method), static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISTypes) - 1))]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "MONOLIS PC type: %s\n", KSPMONOLISPCTypes[std::min(static_cast<PetscInt>(data->prm.precond), static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISPCTypes) - 1))]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetUp_MONOLIS(KSP ksp)
{
  MONOLIS *monolis = (MONOLIS *)ksp->data;
  Mat        A;
  PetscInt   n, nz, ng, Istart, Iend;
  const PetscInt *ghosts;
  PetscInt       *gindex, *myi, *myj;
  
  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQAIJ, &isSeqAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &isMPIAIJ));
  //PetscCall(MatGetLocalSize(A, &n, NULL)); 
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  nloc = Iend - Istart; ng=0;
  if (isMPIAIJ) {
    PetscCall(MatGetGhosts(A, &ng, &ghosts));
  }
  n = nloc+ng;
  //PetscStackCallExternalVoid("Monolis_initialize",monolis_clear(monolis));
  PetscCall(PetscMalloc1(n, &gindex));
  for( PetscInt i=Istart; i<Iend; ++i ) {
	  gindex[i-Istart]=i+1; //std::cout << "gindex:" << i-Istart << "," << gindex[i-Istart] << std::endl;
  }
  //MatView(A, PETSC_VIEWER_STDOUT_WORLD);
  if (isSeqAIJ) {
    Mat_SeqAIJ *aa = (Mat_SeqAIJ *)A->data;
    //std::cout << aa->i[0] << ", "<< aa->i[1] << ", " << aa->i[2] << ", nn=" << aa->nz << std::endl;
    //std::cout << aa->j[0] << ", "<< aa->j[1] << ", " << aa->j[2] << ", " << aa->j[3] << std::endl;
    PetscStackCallExternalVoid("Monolis_set_matrix_BCSR",monolis_set_matrix_BCSR(monolis,n,n,1,aa->nz,aa->a,aa->i,aa->j));
  } else {
    Mat_MPIAIJ        *mat = (Mat_MPIAIJ *)A->data;
    Mat_SeqAIJ        *aa  = (Mat_SeqAIJ *)(mat->A)->data;   // diag part
    Mat_SeqAIJ        *bb  = (Mat_SeqAIJ *)(mat->B)->data;   // off-diag part
    nz = aa->nz + bb->nz;
	  PetscCall(PetscMalloc1(nz, &matv));
    PetscCall(PetscMalloc1(n, &rhs));
    PetscCall(PetscMalloc1(n, &lhs));
    //nd = mat->A->cmap->n;    /* number of columns in diagonal portion */
    for( PetscInt i=0; i<ng; ++i ) {
	    gindex[i+nloc]=ghosts[i]+1;
    }
	PetscCall(PetscMalloc1(nloc+1, &myi));
	PetscCall(PetscMalloc1(nz, &myj));
    myi[0] = 0;
	int cnt_ghost = 0;  // counter of off-diag nonzero components
    for( PetscInt i=0; i<nloc; ++i ) {
      myi[i+1] = myi[i]+ aa->ilen[i] + bb->ilen[i]; 
      for( PetscInt j=aa->i[i]; j<aa->i[i+1]; ++j ) {
        myj[cnt_ghost+j] = aa->j[j];
      }
      int ilast = cnt_ghost+aa->i[i+1];
	  int lcnt = 0;
      for( PetscInt j=bb->i[i]; j<bb->i[i+1]; ++j ) {
        myj[ilast+lcnt] = nloc+bb->j[j];
		++lcnt;
      }
	  cnt_ghost += bb->ilen[i];
    }
    PetscStackCallExternalVoid("Monolis_set_matrix_BCSR",monolis_set_matrix_BCSR(monolis,nloc,n,1,nz,matv,myi,myj));
	PetscCall( PetscFree(myj) );
	PetscCall( PetscFree(myi) );
  }
  PetscStackCallExternalVoid("monolis_com_get_comm_table",monolis_com_get_comm_table(monolis,nloc,n,gindex));
  PetscCall( PetscFree(gindex) );
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_MONOLIS(KSP ksp)
{
  MONOLIS *monolis = (MONOLIS *)ksp->data;

  PetscFunctionBegin;
  if (isMPIAIJ) {
    PetscCall( PetscFree(matv) );
	PetscCall( PetscFree(rhs) );
	PetscCall( PetscFree(lhs) );
  } 
  PetscStackCallExternalVoid("Monolis_finalize",monolis_finalize(monolis));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMONOLISSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMONOLISGetType_C", NULL));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode KSPSolve_MONOLIS_Private(KSP ksp, PetscScalar *b, PetscScalar *x)
{
  Mat                A;
  MONOLIS *monolis = (MONOLIS *)ksp->data;
  int                its;
  double             residual;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  if (isSeqAIJ) {
    Mat_SeqAIJ *aa = (Mat_SeqAIJ *)A->data;
    PetscStackCallExternalVoid("monolis_set_matrix_BCSR_mat_val",monolis_set_matrix_BCSR_mat_val(monolis,1,monolis->mat.NZ,aa->a));
	  PetscStackCallExternalVoid("monolis_solve", monolis_solve(monolis,b,x));
  } else {
    Mat_MPIAIJ        *mat = (Mat_MPIAIJ *)A->data;
    Mat_SeqAIJ        *aa  = (Mat_SeqAIJ *)(mat->A)->data;
    Mat_SeqAIJ        *bb  = (Mat_SeqAIJ *)(mat->B)->data;

    int cnt=-1;
    for( PetscInt i=0; i<nloc; ++i ) {
      for( PetscInt j=aa->i[i]; j<aa->i[i+1]; ++j ) {
        matv[++cnt] = aa->a[j];//std::cout << cnt << ", " << matv[cnt] << "," << j << std::endl;
      }
      for( PetscInt j=bb->i[i]; j<bb->i[i+1]; ++j ) {
        matv[++cnt] = bb->a[j];//std::cout << cnt << ",a " << matv[cnt] << "," << j << std::endl;
      }
      rhs[i] = b[i];   lhs[i] = x[i];
    }
    PetscStackCallExternalVoid("monolis_set_matrix_BCSR_mat_val",monolis_set_matrix_BCSR_mat_val(monolis,1,monolis->mat.NZ,matv));
	  PetscStackCallExternalVoid("monolis_solve", monolis_solve(monolis,rhs,lhs));
	  for( PetscInt i=0; i<nloc; ++i ) {
      x[i]= lhs[i];
    }
  }

  PetscStackCallExternalVoid("monolis_get_converge_residual" , monolis_get_converge_residual(monolis, &residual));
  if( residual<monolis->prm.tol )
    ksp->reason = KSP_CONVERGED_ATOL;
  else
    ksp->reason = KSP_DIVERGED_ITS;
  PetscStackCallExternalVoid("monolis_get_converge_iter" , monolis_get_converge_iter(monolis, &its));
  ksp->its = PetscMin(its, ksp->max_it);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_MONOLIS(KSP ksp)
{
  Mat                A;
  PetscScalar       *x;
  PetscScalar *b;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(VecGetArrayWrite(ksp->vec_sol, &x));
  PetscCall(VecGetArray(ksp->vec_rhs, &b));
  PetscCall(KSPSolve_MONOLIS_Private(ksp, b, x));

  PetscCall(VecRestoreArray(ksp->vec_rhs, &b));
  PetscCall(VecRestoreArrayWrite(ksp->vec_sol, &x));
  PetscFunctionReturn(0);
}

/*@
     KSPMONOLISSetType - Sets the type of Krylov method used in `KSPMONOLIS`.

   Collective

   Input Parameters:
+     ksp - iterative context
-     type - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly

   Level: intermediate

   Notes:
     Unlike `KSPReset()`, this function does not destroy any deflation space attached to the `KSP`.

     As an example, in the following sequence:
.vb
     KSPMONOLISSetType(ksp, KSPGCRODR);
     KSPSolve(ksp, b, x);
     KSPMONOLISSetType(ksp, KSPGMRES);
     KSPMONOLISSetType(ksp, KSPGCRODR);
     KSPSolve(ksp, b, x);
.ve
    the recycled space is reused in the second `KSPSolve()`.

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPType`, `KSPMONOLISType`, `KSPMONOLISGetType()`
@*/
PetscErrorCode KSPMONOLISSetType(KSP ksp, KSPMONOLISType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ksp, type, 2);
  PetscUseMethod(ksp, "KSPMONOLISSetType_C", (KSP, KSPMONOLISType), (ksp, type));
  PetscFunctionReturn(0);
}

/*@
   KSPMONOLISGetType - Gets the type of Krylov method used in `KSPMONOLIS`.

   Input Parameter:
.     ksp - iterative context

   Output Parameter:
.     type - any of gmres, bgmres, cg, bcg, gcrodr, bgcrodr, bfbcg, or preonly

   Level: intermediate

.seealso: [](chapter_ksp), `KSPCreate()`, `KSPType`, `KSPMONOLISType`, `KSPMONOLISSetType()`
@*/
PetscErrorCode KSPMONOLISGetType(KSP ksp, KSPMONOLISType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  if (type) {
    PetscValidPointer(type, 2);
    PetscUseMethod(ksp, "KSPMONOLISGetType_C", (KSP, KSPMONOLISType *), (ksp, type));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMONOLISSetType_MONOLIS(KSP ksp, KSPMONOLISType type)
{
  MONOLIS *data = (MONOLIS *)ksp->data;
  PetscInt   i;
  PetscBool  flg = PETSC_FALSE;

  PetscFunctionBegin;
  for (i = 0; i < static_cast<PetscInt>(PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISTypes)); ++i) {
    PetscCall(PetscStrcmp(KSPMONOLISTypes[type], KSPMONOLISTypes[i], &flg));
    if (flg) break;
  }
  PetscCheck(i != PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISTypes), PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown KSPMONOLISType %d", type);
  //if (data->cntl[0] != static_cast<char>(PETSC_DECIDE) && data->cntl[0] != i) PetscCall(KSPHPDDMReset_Private(ksp));
  data->prm.method = i;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPMONOLISGetType_MONOLIS(KSP ksp, KSPMONOLISType *type)
{
  MONOLIS *data = (MONOLIS *)ksp->data;

  PetscFunctionBegin;
  PetscCheck(data->prm.method <= 0, PETSC_COMM_SELF, PETSC_ERR_ORDER, "KSPMONOLISType not set yet");
  /* need to shift by -1 for HPDDM_KRYLOV_METHOD_NONE */
  *type = static_cast<KSPMONOLISType>(PetscMin(data->prm.method, static_cast<char>(PETSC_STATIC_ARRAY_LENGTH(KSPMONOLISTypes) - 1)));
  PetscFunctionReturn(0);
}


/*MC
     KSPMONOLIS - Interface with the MONOLIS library. 
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_MONOLIS(KSP ksp)
{
  MONOLIS  *data;

  PetscFunctionBegin;
  PetscCall(PetscNew(&data));
  ksp->data = (void *)data;
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 1));
  ksp->ops->solve          = KSPSolve_MONOLIS;
  ksp->ops->setup          = KSPSetUp_MONOLIS;
  ksp->ops->setfromoptions = KSPSetFromOptions_MONOLIS;
  ksp->ops->destroy        = KSPDestroy_MONOLIS;
  ksp->ops->view           = KSPView_MONOLIS;
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMONOLISSetType_C", KSPMONOLISSetType_MONOLIS));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPMONOLISGetType_C", KSPMONOLISGetType_MONOLIS));
  PetscFunctionReturn(0);
}

