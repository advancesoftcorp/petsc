#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DAGetGlobalIndices"
/*@C
   DAGetGlobalIndices - Returns the global node number of all local nodes,
   including ghost nodes.

   Not Collective

   Input Parameter:
.  da - the distributed array

   Output Parameters:
+  n - the number of local elements, including ghost nodes (or PETSC_NULL)
-  idx - the global indices

   Level: intermediate

   Note: 
   For DA_STENCIL_STAR stencils the inactive corner ghost nodes are also included
   in the list of local indices (even though those nodes are not updated 
   during calls to DAXXXToXXX().

   Essentially the same data is returned in the form of a local-to-global mapping
   with the routine DAGetISLocalToGlobalMapping();

   Fortran Note:
   This routine is used differently from Fortran
.vb
        DA          da
        integer     n,da_array(1)
        PetscOffset i_da
        integer     ierr
        call DAGetGlobalIndices(da,n,da_array,i_da,ierr)

   C Access first local entry in list
        value = da_array(i_da + 1)
.ve

   See the <A href="../../docs/manual.pdf#nameddest=ch_fortran">Fortran chapter</A> of the users manual for details.

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlobal()
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DALocalToLocalBegin(), DAGetAO(), DAGetGlobalIndicesF90()
          DAGetISLocalToGlobalMapping(), DACreate3d(), DACreate1d(), DALocalToLocalEnd(), DAGetOwnershipRanges()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetGlobalIndices(DA da,PetscInt *n,PetscInt **idx)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (n)   *n   = dd->Nl;
  if (idx) *idx = dd->idx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetNatural_Private"
/*
   Gets the natural number for each global number on the process.

   Used by DAGetAO() and DAGlobalToNatural_Create()
*/
PetscErrorCode DAGetNatural_Private(DA da,PetscInt *outNlocal,IS *isnatural)
{
  PetscErrorCode ierr;
  PetscInt       Nlocal,i,j,k,*lidx,lict = 0;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  Nlocal = (dd->xe-dd->xs);
  if (dd->dim > 1) {
    Nlocal *= (dd->ye-dd->ys);
  } 
  if (dd->dim > 2) {
    Nlocal *= (dd->ze-dd->zs);
  }
  
  ierr = PetscMalloc(Nlocal*sizeof(PetscInt),&lidx);CHKERRQ(ierr);
  
  if (dd->dim == 1) {
    for (i=dd->xs; i<dd->xe; i++) {
      /*  global number in natural ordering */
      lidx[lict++] = i;
    }
  } else if (dd->dim == 2) {
    for (j=dd->ys; j<dd->ye; j++) {
      for (i=dd->xs; i<dd->xe; i++) {
	/*  global number in natural ordering */
	lidx[lict++] = i + j*dd->M*dd->w;
      }
    }
  } else if (dd->dim == 3) {
    for (k=dd->zs; k<dd->ze; k++) {
      for (j=dd->ys; j<dd->ye; j++) {
	for (i=dd->xs; i<dd->xe; i++) {
	  lidx[lict++] = i + j*dd->M*dd->w + k*dd->M*dd->N*dd->w;
	}
      }
    }
  }
  *outNlocal = Nlocal;
  ierr = ISCreateGeneral(((PetscObject)da)->comm,Nlocal,lidx,PETSC_OWN_POINTER,isnatural);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetAO"
/*@
   DAGetAO - Gets the application ordering context for a distributed array.

   Collective on DA

   Input Parameter:
.  da - the distributed array

   Output Parameters:
.  ao - the application ordering context for DAs

   Level: intermediate

   Notes:
   In this case, the AO maps to the natural grid ordering that would be used
   for the DA if only 1 processor were employed (ordering most rapidly in the
   x-direction, then y, then z).  Multiple degrees of freedom are numbered
   for each node (rather than 1 component for the whole grid, then the next
   component, etc.)

.keywords: distributed array, get, global, indices, local-to-global

.seealso: DACreate2d(), DAGetGhostCorners(), DAGetCorners(), DALocalToGlocal()
          DAGlobalToLocalBegin(), DAGlobalToLocalEnd(), DALocalToLocalBegin(), DALocalToLocalEnd(), DAGetGlobalIndices(), DAGetOwnershipRanges(),
          AO, AOPetscToApplication(), AOApplicationToPetsc()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DAGetAO(DA da,AO *ao)
{
  DM_DA *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidPointer(ao,2);

  /* 
     Build the natural ordering to PETSc ordering mappings.
  */
  if (!dd->ao) {
    IS             ispetsc,isnatural;
    PetscErrorCode ierr;
    PetscInt       Nlocal;

    ierr = DAGetNatural_Private(da,&Nlocal,&isnatural);CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)da)->comm,Nlocal,dd->base,1,&ispetsc);CHKERRQ(ierr);
    ierr = AOCreateBasicIS(isnatural,ispetsc,&dd->ao);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(da,dd->ao);CHKERRQ(ierr);
    ierr = ISDestroy(ispetsc);CHKERRQ(ierr);
    ierr = ISDestroy(isnatural);CHKERRQ(ierr);
  }
  *ao = dd->ao;
  PetscFunctionReturn(0);
}

/*MC
    DAGetGlobalIndicesF90 - Returns a Fortran90 pointer to the list of 
    global indices (global node number of all local nodes, including
    ghost nodes).

    Synopsis:
    DAGetGlobalIndicesF90(DA da,integer n,{integer, pointer :: idx(:)},integer ierr)

    Not Collective

    Input Parameter:
.   da - the distributed array

    Output Parameters:
+   n - the number of local elements, including ghost nodes (or PETSC_NULL)
.   idx - the Fortran90 pointer to the global indices
-   ierr - error code

    Level: intermediate

    Notes:
     Not yet supported for all F90 compilers

.keywords: distributed array, get, global, indices, local-to-global, f90

.seealso: DAGetGlobalIndices()
M*/
