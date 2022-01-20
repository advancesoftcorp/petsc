/*
   Functions specific to the 1-dimensional implementation of DMStag
*/
#include <petsc/private/dmstagimpl.h>

/*@C
  DMStagCreate1d - Create an object to manage data living on the elements and vertices of a parallelized regular 1D grid.

  Collective

  Input Parameters:
+ comm - MPI communicator
. bndx - boundary type: DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, or DM_BOUNDARY_GHOSTED
. M - global number of grid points
. dof0 - number of degrees of freedom per vertex/0-cell
. dof1 - number of degrees of freedom per element/1-cell
. stencilType - ghost/halo region type: DMSTAG_STENCIL_BOX or DMSTAG_STENCIL_NONE
. stencilWidth - width, in elements, of halo/ghost region
- lx - array of local sizes, of length equal to the comm size, summing to M

  Output Parameter:
. dm - the new DMStag object

  Options Database Keys:
+ -dm_view - calls DMViewFromOptions() a the conclusion of DMSetUp()
. -stag_grid_x <nx> - number of elements in the x direction
. -stag_ghost_stencil_width - width of ghost region, in elements
- -stag_boundary_type_x <none,ghosted,periodic> - DMBoundaryType value

  Notes:
  You must call DMSetUp() after this call before using the DM.
  If you wish to use the options database (see the keys above) to change values in the DMStag, you must call
  DMSetFromOptions() after this function but before DMSetUp().

  Level: beginner

.seealso: DMSTAG, DMStagCreate2d(), DMStagCreate3d(), DMDestroy(), DMView(), DMCreateGlobalVector(), DMCreateLocalVector(), DMLocalToGlobalBegin(), DMDACreate1d()
@*/
PETSC_EXTERN PetscErrorCode DMStagCreate1d(MPI_Comm comm,DMBoundaryType bndx,PetscInt M,PetscInt dof0,PetscInt dof1,DMStagStencilType stencilType,PetscInt stencilWidth,const PetscInt lx[],DM* dm)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = DMCreate(comm,dm);CHKERRQ(ierr);
  ierr = DMSetDimension(*dm,1);CHKERRQ(ierr);
  ierr = DMStagInitialize(bndx,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,M,0,0,size,0,0,dof0,dof1,0,0,stencilType,stencilWidth,lx,NULL,NULL,*dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagSetUniformCoordinatesExplicit_1d(DM dm,PetscReal xmin,PetscReal xmax)
{
  PetscErrorCode ierr;
  DM_Stag        *stagCoord;
  DM             dmCoord;
  Vec            coordLocal;
  PetscReal      h,min;
  PetscScalar    **arr;
  PetscInt       start_ghost,n_ghost,s;
  PetscInt       ileft,ielement;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  stagCoord = (DM_Stag*) dmCoord->data;
  for (s=0; s<2; ++s) {
    if (stagCoord->dof[s] !=0 && stagCoord->dof[s] != 1) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Coordinate DM in 1 dimensions must have 0 or 1 dof on each stratum, but stratum %d has %d dof",s,stagCoord->dof[s]);
  }
  ierr = DMCreateLocalVector(dmCoord,&coordLocal);CHKERRQ(ierr);

  ierr = DMStagVecGetArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  if (stagCoord->dof[0]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_LEFT,0,&ileft);CHKERRQ(ierr);
  }
  if (stagCoord->dof[1]) {
    ierr = DMStagGetLocationSlot(dmCoord,DMSTAG_ELEMENT,0,&ielement);CHKERRQ(ierr);
  }
  ierr = DMStagGetGhostCorners(dmCoord,&start_ghost,NULL,NULL,&n_ghost,NULL,NULL);CHKERRQ(ierr);

  min = xmin;
  h = (xmax-xmin)/stagCoord->N[0];

  for (PetscInt ind=start_ghost; ind<start_ghost + n_ghost; ++ind) {
    if (stagCoord->dof[0]) {
      const PetscReal off = 0.0;
        arr[ind][ileft] = min + ((PetscReal)ind + off) * h;
    }
    if (stagCoord->dof[1]) {
      const PetscReal off = 0.5;
        arr[ind][ielement] = min + ((PetscReal)ind + off) * h;
    }
  }
  ierr = DMStagVecRestoreArray(dmCoord,coordLocal,&arr);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(dm,coordLocal);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)coordLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&coordLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Helper functions used in DMSetUp_Stag() */
static PetscErrorCode DMStagComputeLocationOffsets_1d(DM);

PETSC_INTERN PetscErrorCode DMSetUp_Stag_1d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscMPIInt     size,rank;
  MPI_Comm        comm;
  PetscInt        j;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  /* Check Global size */
  if (stag->N[0] < 1) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Global grid size of %D < 1 specified",stag->N[0]);

  /* Local sizes */
  if (stag->N[0] < size) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"More ranks (%d) than elements (%D) specified",size,stag->N[0]);
  if (!stag->l[0]) {
    /* Divide equally, giving an extra elements to higher ranks */
    ierr = PetscMalloc1(stag->nRanks[0],&stag->l[0]);CHKERRQ(ierr);
    for (j=0; j<stag->nRanks[0]; ++j) stag->l[0][j] = stag->N[0]/stag->nRanks[0] + (stag->N[0] % stag->nRanks[0] > j ? 1 : 0);
  }
  {
    PetscInt Nchk = 0;
    for (j=0; j<size; ++j) Nchk += stag->l[0][j];
    if (Nchk != stag->N[0]) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"Sum of specified local sizes (%D) is not equal to global size (%D)",Nchk,stag->N[0]);
  }
  stag->n[0] = stag->l[0][rank];

  /* Rank (trivial in 1d) */
  stag->rank[0]      = rank;
  stag->firstRank[0] = (PetscBool)(rank == 0);
  stag->lastRank[0]  = (PetscBool)(rank == size-1);

  /* Local (unghosted) numbers of entries */
  stag->entriesPerElement = stag->dof[0] + stag->dof[1];
  switch (stag->boundaryType[0]) {
    case DM_BOUNDARY_NONE:
    case DM_BOUNDARY_GHOSTED:  stag->entries = stag->n[0] * stag->entriesPerElement + (stag->lastRank[0] ?  stag->dof[0] : 0); break;
    case DM_BOUNDARY_PERIODIC: stag->entries = stag->n[0] * stag->entriesPerElement;                                           break;
    default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
  }

  /* Starting element */
  stag->start[0] = 0;
  for (j=0; j<stag->rank[0]; ++j) stag->start[0] += stag->l[0][j];

  /* Local/ghosted size and starting element */
  switch (stag->boundaryType[0]) {
    case DM_BOUNDARY_NONE :
      switch (stag->stencilType) {
        case DMSTAG_STENCIL_NONE : /* Only dummy cells on the right */
          stag->startGhost[0] = stag->start[0];
          stag->nGhost[0]     = stag->n[0] + (stag->lastRank[0] ? 1 : 0);
          break;
        case DMSTAG_STENCIL_STAR :
        case DMSTAG_STENCIL_BOX :
          stag->startGhost[0] = stag->firstRank[0] ? stag->start[0]: stag->start[0] - stag->stencilWidth;
          stag->nGhost[0] = stag->n[0];
          stag->nGhost[0] += stag->firstRank[0] ? 0 : stag->stencilWidth;
          stag->nGhost[0] += stag->lastRank[0]  ? 1 : stag->stencilWidth;
          break;
        default :
          SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    case DM_BOUNDARY_GHOSTED:
      switch (stag->stencilType) {
        case DMSTAG_STENCIL_NONE :
          stag->startGhost[0] = stag->start[0];
          stag->nGhost[0]     = stag->n[0] + (stag->lastRank[0] ? 1 : 0);
          break;
        case DMSTAG_STENCIL_STAR :
        case DMSTAG_STENCIL_BOX :
          stag->startGhost[0] = stag->start[0] - stag->stencilWidth; /* This value may be negative */
          stag->nGhost[0]     = stag->n[0] + 2*stag->stencilWidth + (stag->lastRank[0] && stag->stencilWidth == 0 ? 1 : 0);
          break;
        default :
          SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    case DM_BOUNDARY_PERIODIC:
      switch (stag->stencilType) {
        case DMSTAG_STENCIL_NONE :
          stag->startGhost[0] = stag->start[0];
          stag->nGhost[0]     = stag->n[0];
          break;
        case DMSTAG_STENCIL_STAR :
        case DMSTAG_STENCIL_BOX :
          stag->startGhost[0] = stag->start[0] - stag->stencilWidth; /* This value may be negative */
          stag->nGhost[0]     = stag->n[0] + 2*stag->stencilWidth;
          break;
        default :
          SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unrecognized ghost stencil type %d",stag->stencilType);
      }
      break;
    default :
      SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
  }

  /* Total size of ghosted/local representation */
  stag->entriesGhost = stag->nGhost[0]*stag->entriesPerElement;

  /* Define neighbors */
  ierr = PetscMalloc1(3,&stag->neighbors);CHKERRQ(ierr);
  if (stag->firstRank[0]) {
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:     stag->neighbors[0] = -1;                break;
      case DM_BOUNDARY_PERIODIC: stag->neighbors[0] = stag->nRanks[0]-1; break;
      default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }
  } else {
    stag->neighbors[0] = stag->rank[0]-1;
  }
  stag->neighbors[1] = stag->rank[0];
  if (stag->lastRank[0]) {
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:     stag->neighbors[2] = -1;                break;
      case DM_BOUNDARY_PERIODIC: stag->neighbors[2] = 0;                 break;
      default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }
  } else {
    stag->neighbors[2] = stag->rank[0]+1;
  }

  if (stag->n[0] < stag->stencilWidth) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"DMStag 1d setup does not support local sizes (%d) smaller than the elementwise stencil width (%d)",stag->n[0],stag->stencilWidth);
  }

  /* Create global->local VecScatter and ISLocalToGlobalMapping */
  {
    PetscInt *idxLocal,*idxGlobal,*idxGlobalAll;
    PetscInt i,iLocal,d,entriesToTransferTotal,ghostOffsetStart,ghostOffsetEnd,nNonDummyGhost;
    IS       isLocal,isGlobal;

    /* The offset on the right (may not be equal to the stencil width, as we
       always have at least one ghost element, to account for the boundary
       point, and may with ghosted boundaries), and the number of non-dummy ghost elements */
    ghostOffsetStart = stag->start[0] - stag->startGhost[0];
    ghostOffsetEnd   = stag->startGhost[0]+stag->nGhost[0] - (stag->start[0]+stag->n[0]);
    nNonDummyGhost   = stag->nGhost[0] - (stag->lastRank[0] ? ghostOffsetEnd : 0) - (stag->firstRank[0] ? ghostOffsetStart : 0);

    /* Compute the number of non-dummy entries in the local representation
       This is equal to the number of non-dummy elements in the local (ghosted) representation,
       plus some extra entries on the right boundary on the last rank*/
    switch (stag->boundaryType[0]) {
      case DM_BOUNDARY_GHOSTED:
      case DM_BOUNDARY_NONE:
        entriesToTransferTotal = nNonDummyGhost * stag->entriesPerElement + (stag->lastRank[0] ? stag->dof[0] : 0);
        break;
      case DM_BOUNDARY_PERIODIC:
        entriesToTransferTotal = stag->entriesGhost; /* No dummy points */
        break;
      default :
        SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);
    }

    ierr = PetscMalloc1(entriesToTransferTotal,&idxLocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(entriesToTransferTotal,&idxGlobal);CHKERRQ(ierr);
    ierr = PetscMalloc1(stag->entriesGhost,&idxGlobalAll);CHKERRQ(ierr);
    if (stag->boundaryType[0] == DM_BOUNDARY_NONE) {
      PetscInt count = 0,countAll = 0;
      /* Left ghost points and native points */
      for (i=stag->startGhost[0], iLocal=0; iLocal<nNonDummyGhost; ++i,++iLocal) {
        for (d=0; d<stag->entriesPerElement; ++d,++count,++countAll) {
          idxLocal [count]       = iLocal * stag->entriesPerElement + d;
          idxGlobal[count]       = i      * stag->entriesPerElement + d;
          idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
        }
      }
      /* Ghost points on the right
         Special case for last (partial dummy) element on the last rank */
      if (stag->lastRank[0]) {
        i      = stag->N[0];
        iLocal = (stag->nGhost[0]-ghostOffsetEnd);
        /* Only vertex (0-cell) dofs in global representation */
        for (d=0; d<stag->dof[0]; ++d,++count,++countAll) {
          idxGlobal[count]       = i      * stag->entriesPerElement + d;
          idxLocal [count]       = iLocal * stag->entriesPerElement + d;
          idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
        }
        for (d=stag->dof[0]; d<stag->entriesPerElement; ++d,++countAll) { /* Additional dummy entries */
          idxGlobalAll[countAll] = -1;
        }
      }
    } else if (stag->boundaryType[0] == DM_BOUNDARY_PERIODIC) {
      PetscInt count = 0,iLocal = 0; /* No dummy points, so idxGlobal and idxGlobalAll are identical */
      const PetscInt iMin = stag->firstRank[0] ? stag->start[0] : stag->startGhost[0];
      const PetscInt iMax = stag->lastRank[0] ? stag->startGhost[0] + stag->nGhost[0] - stag->stencilWidth : stag->startGhost[0] + stag->nGhost[0];
      /* Ghost points on the left */
      if (stag->firstRank[0]) {
        for (i=stag->N[0]-stag->stencilWidth; iLocal<stag->stencilWidth; ++i,++iLocal) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobal[count] = i      * stag->entriesPerElement + d;
            idxLocal [count] = iLocal * stag->entriesPerElement + d;
            idxGlobalAll[count] = idxGlobal[count];
          }
        }
      }
      /* Native points */
      for (i=iMin; i<iMax; ++i,++iLocal) {
        for (d=0; d<stag->entriesPerElement; ++d,++count) {
          idxGlobal[count] = i      * stag->entriesPerElement + d;
          idxLocal [count] = iLocal * stag->entriesPerElement + d;
          idxGlobalAll[count] = idxGlobal[count];
        }
      }
      /* Ghost points on the right */
      if (stag->lastRank[0]) {
        for (i=0; iLocal<stag->nGhost[0]; ++i,++iLocal) {
          for (d=0; d<stag->entriesPerElement; ++d,++count) {
            idxGlobal[count] = i      * stag->entriesPerElement + d;
            idxLocal [count] = iLocal * stag->entriesPerElement + d;
            idxGlobalAll[count] = idxGlobal[count];
          }
        }
      }
    } else if (stag->boundaryType[0] == DM_BOUNDARY_GHOSTED) {
      PetscInt count = 0,countAll = 0;
      /* Dummy elements on the left, on the first rank */
      if (stag->firstRank[0]) {
        for (iLocal=0; iLocal<ghostOffsetStart; ++iLocal) {
          /* Complete elements full of dummy entries */
          for (d=0; d<stag->entriesPerElement; ++d,++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
        i = 0; /* nonDummy entries start with global entry 0 */
      } else {
        /* nonDummy entries start as usual */
        i = stag->startGhost[0];
        iLocal = 0;
      }

      /* non-Dummy entries */
      {
        PetscInt iLocalNonDummyMax = stag->firstRank[0] ? nNonDummyGhost + ghostOffsetStart : nNonDummyGhost;
        for (; iLocal<iLocalNonDummyMax; ++i,++iLocal) {
          for (d=0; d<stag->entriesPerElement; ++d,++count,++countAll) {
            idxLocal [count]       = iLocal * stag->entriesPerElement + d;
            idxGlobal[count]       = i      * stag->entriesPerElement + d;
            idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
          }
        }
      }

      /* (partial) dummy elements on the right, on the last rank */
      if (stag->lastRank[0]) {
        /* First one is partial dummy */
        i      = stag->N[0];
        iLocal = (stag->nGhost[0]-ghostOffsetEnd);
        for (d=0; d<stag->dof[0]; ++d,++count,++countAll) { /* Only vertex (0-cell) dofs in global representation */
          idxLocal [count]       = iLocal * stag->entriesPerElement + d;
          idxGlobal[count]       = i      * stag->entriesPerElement + d;
          idxGlobalAll[countAll] = i      * stag->entriesPerElement + d;
        }
        for (d=stag->dof[0]; d<stag->entriesPerElement; ++d,++countAll) { /* Additional dummy entries */
          idxGlobalAll[countAll] = -1;
        }
        for (iLocal = stag->nGhost[0] - ghostOffsetEnd + 1; iLocal < stag->nGhost[0]; ++iLocal) {
          /* Additional dummy elements */
          for (d=0; d<stag->entriesPerElement; ++d,++countAll) {
            idxGlobalAll[countAll] = -1;
          }
        }
      }
    } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported x boundary type %s",DMBoundaryTypes[stag->boundaryType[0]]);

    /* Create Local IS (transferring pointer ownership) */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxLocal,PETSC_OWN_POINTER,&isLocal);CHKERRQ(ierr);

    /* Create Global IS (transferring pointer ownership) */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),entriesToTransferTotal,idxGlobal,PETSC_OWN_POINTER,&isGlobal);CHKERRQ(ierr);

    /* Create stag->gtol, which doesn't include dummy entries */
    {
      Vec local,global;
      ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local);CHKERRQ(ierr);
      ierr = VecScatterCreate(global,isGlobal,local,isLocal,&stag->gtol);CHKERRQ(ierr);
      ierr = VecDestroy(&global);CHKERRQ(ierr);
      ierr = VecDestroy(&local);CHKERRQ(ierr);
    }

    /* In special cases, create a dedicated injective local-to-global map */
    if (stag->boundaryType[0] == DM_BOUNDARY_PERIODIC && stag->nRanks[0] == 1) {
      ierr = DMStagPopulateLocalToGlobalInjective(dm);CHKERRQ(ierr);
    }

    /* Destroy ISs */
    ierr = ISDestroy(&isLocal);CHKERRQ(ierr);
    ierr = ISDestroy(&isGlobal);CHKERRQ(ierr);

    /* Create local-to-global map (transferring pointer ownership) */
    ierr = ISLocalToGlobalMappingCreate(comm,1,stag->entriesGhost,idxGlobalAll,PETSC_OWN_POINTER,&dm->ltogmap);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)dm,(PetscObject)dm->ltogmap);CHKERRQ(ierr);
  }

  /* Precompute location offsets */
  ierr = DMStagComputeLocationOffsets_1d(dm);CHKERRQ(ierr);

  /* View from Options */
  ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);

 PetscFunctionReturn(0);
}

static PetscErrorCode DMStagComputeLocationOffsets_1d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt  epe = stag->entriesPerElement;

  PetscFunctionBegin;
  ierr = PetscMalloc1(DMSTAG_NUMBER_LOCATIONS,&stag->locationOffsets);CHKERRQ(ierr);
  stag->locationOffsets[DMSTAG_LEFT]    = 0;
  stag->locationOffsets[DMSTAG_ELEMENT] = stag->locationOffsets[DMSTAG_LEFT] + stag->dof[0];
  stag->locationOffsets[DMSTAG_RIGHT]   = stag->locationOffsets[DMSTAG_LEFT] + epe;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMStagPopulateLocalToGlobalInjective_1d(DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        *idxLocal,*idxGlobal;
  PetscInt        i,iLocal,d,count;
  IS              isLocal,isGlobal;

  PetscFunctionBegin;
  ierr = PetscMalloc1(stag->entries,&idxLocal);CHKERRQ(ierr);
  ierr = PetscMalloc1(stag->entries,&idxGlobal);CHKERRQ(ierr);
  count = 0;
  iLocal = stag->start[0]-stag->startGhost[0];
  for (i=stag->start[0]; i<stag->start[0]+stag->n[0]; ++i,++iLocal) {
    for (d=0; d<stag->entriesPerElement; ++d,++count) {
      idxGlobal[count] = i      * stag->entriesPerElement + d;
      idxLocal [count] = iLocal * stag->entriesPerElement + d;
    }
  }
  if (stag->lastRank[0] && stag->boundaryType[0] != DM_BOUNDARY_PERIODIC) {
    i = stag->start[0]+stag->n[0];
    iLocal = stag->start[0]-stag->startGhost[0] + stag->n[0];
    for (d=0; d<stag->dof[0]; ++d,++count) {
      idxGlobal[count] = i      * stag->entriesPerElement + d;
      idxLocal [count] = iLocal * stag->entriesPerElement + d;
    }
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),stag->entries,idxLocal,PETSC_OWN_POINTER,&isLocal);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm),stag->entries,idxGlobal,PETSC_OWN_POINTER,&isGlobal);CHKERRQ(ierr);
  {
    Vec local,global;
    ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)dm),1,stag->entries,PETSC_DECIDE,NULL,&global);CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,stag->entriesPerElement,stag->entriesGhost,NULL,&local);CHKERRQ(ierr);
    ierr = VecScatterCreate(local,isLocal,global,isGlobal,&stag->ltog_injective);CHKERRQ(ierr);
    ierr = VecDestroy(&global);CHKERRQ(ierr);
    ierr = VecDestroy(&local);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&isLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&isGlobal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMCreateMatrix_Stag_1D_AIJ(DM dm,Mat *mat)
{
  PetscErrorCode         ierr;
  PetscInt               entries,dof[DMSTAG_MAX_STRATA],epe,stencil_width,max_nz_per_row,N,start,n,n_extra;
  DMStagStencilType      stencil_type;
  ISLocalToGlobalMapping ltogmap;
  DMBoundaryType         boundary_type_x;

  /* This implementation gives a very dense stencil, which is likely unsuitable for
     (typical) applications which have fewer couplings */
  PetscFunctionBegin;
  ierr = DMStagGetDOF(dm,&dof[0],&dof[1],NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetStencilType(dm,&stencil_type);CHKERRQ(ierr);
  ierr = DMStagGetStencilWidth(dm,&stencil_width);CHKERRQ(ierr);
  ierr = DMStagGetEntries(dm,&entries);CHKERRQ(ierr);
  ierr = DMStagGetEntriesPerElement(dm,&epe);CHKERRQ(ierr);
  ierr = DMStagGetCorners(dm,&start,NULL,NULL,&n,NULL,NULL,&n_extra,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetGlobalSizes(dm,&N,NULL,NULL);CHKERRQ(ierr);
  ierr = DMStagGetBoundaryTypes(dm,&boundary_type_x,NULL,NULL);CHKERRQ(ierr);

  if (stencil_type == DMSTAG_STENCIL_NONE) {
    max_nz_per_row = PetscMax(dof[0],dof[1]);CHKERRQ(ierr);
  } else if (stencil_type == DMSTAG_STENCIL_STAR || stencil_type == DMSTAG_STENCIL_BOX) {
    max_nz_per_row = (1 + 2 * stencil_width) * epe;
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported stencil type %s",DMStagStencilTypes[stencil_type]);
  ierr = MatCreateAIJ(PetscObjectComm((PetscObject)dm),entries,entries,PETSC_DETERMINE,PETSC_DETERMINE,max_nz_per_row,NULL,max_nz_per_row,NULL,mat);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm,&ltogmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*mat,ltogmap,ltogmap);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*mat);CHKERRQ(ierr);

  if (!dm->prealloc_only) {
    if (stencil_type == DMSTAG_STENCIL_NONE) {
      /* Couple all DOF at each location to each other */
      DMStagStencil row_vertex,row_element;
      DMStagStencil *col_vertex,*col_element;

      row_vertex.loc = DMSTAG_LEFT;
      ierr = PetscMalloc1(dof[0],&col_vertex);CHKERRQ(ierr);
      for (PetscInt c2=0; c2<dof[0]; ++c2) {
        col_vertex[c2].loc = DMSTAG_LEFT;
      }

      row_element.loc = DMSTAG_ELEMENT;
      ierr = PetscMalloc1(dof[1],&col_element);CHKERRQ(ierr);
      for (PetscInt c2=0; c2<dof[1]; ++c2) {
        col_element[c2].loc = DMSTAG_ELEMENT;
      }

      for (PetscInt e=start; e<start+n+n_extra; ++e) {
        {
          row_vertex.i = e;
          for (PetscInt c=0; c<dof[0]; ++c) {
            row_vertex.c = c;
            for (PetscInt c2=0; c2<dof[0]; ++c2){
              col_vertex[c2].i = e;
              col_vertex[c2].c = c2;
            }
            ierr = DMStagMatSetValuesStencil(dm,*mat,1,&row_vertex,dof[0],col_vertex,NULL,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
        if (e < N) {
          row_element.i = e;
          for (PetscInt c=0; c<dof[1]; ++c) {
            row_element.c = c;
            for (PetscInt c2=0; c2<dof[1]; ++c2) {
              col_element[c2].i = e;
              col_element[c2].c = c2;
            }
            ierr = DMStagMatSetValuesStencil(dm,*mat,1,&row_element,dof[1],col_element,NULL,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
      ierr = PetscFree(col_vertex);CHKERRQ(ierr);
      ierr = PetscFree(col_element);CHKERRQ(ierr);
    } else if (stencil_type == DMSTAG_STENCIL_STAR || stencil_type == DMSTAG_STENCIL_BOX) {
      DMStagStencil *col,*row;

      ierr = PetscMalloc1(epe,&row);CHKERRQ(ierr);
      {
        PetscInt nrows = 0;
        for (PetscInt c=0; c<dof[0]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_LEFT;
          ++nrows;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          row[nrows].c = c;
          row[nrows].loc = DMSTAG_ELEMENT;
          ++nrows;
        }
      }
      max_nz_per_row = (1 + 2 * stencil_width) * epe;
      ierr = PetscMalloc1(epe,&col);CHKERRQ(ierr);
      {
        PetscInt ncols = 0;
        for (PetscInt c=0; c<dof[0]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_LEFT;
          ++ncols;
        }
        for (PetscInt c=0; c<dof[1]; ++c) {
          col[ncols].c = c;
          col[ncols].loc = DMSTAG_ELEMENT;
          ++ncols;
        }
      }
      for (PetscInt e=start; e<start+n+n_extra; ++e) {
        for (PetscInt i=0; i<epe; ++i) {
          row[i].i = e;
        }
        for (PetscInt offset = -stencil_width; offset<=stencil_width; ++offset) {
          const PetscInt e_offset = e + offset;

          /* Only set values corresponding to elements which can have non-dummy entries,
             meaning those that map to unknowns in the global representation. In the periodic
             case, this is the entire stencil, but in all other cases, only includes a single
             "extra" element which is partially outside the physical domain (those points in the
             global representation */
          if (boundary_type_x == DM_BOUNDARY_PERIODIC || (e_offset < N+1 && e_offset >= 0)) {
            for (PetscInt i=0; i<epe; ++i) {
              col[i].i = e_offset;
            }
            ierr = DMStagMatSetValuesStencil(dm,*mat,epe,row,epe,col,NULL,INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
      ierr = PetscFree(row);CHKERRQ(ierr);
      ierr = PetscFree(col);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported stencil type %s",DMStagStencilTypes[stencil_type]);
    ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Note: GPU-related logic, e.g. at the end of DMCreateMatrix_DA_1d_MPIAIJ, is not included here
       but might be desirable */
  }
  PetscFunctionReturn(0);
}
