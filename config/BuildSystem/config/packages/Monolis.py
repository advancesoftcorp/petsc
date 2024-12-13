import config.package

class Configure(config.package.Package):
  def __init__(self, framework):
    config.package.Package.__init__(self, framework)
    self.version          = '1.0.1'
    self.minversion       = '1.0.1'
    self.versionname      = 'MONOLIS_VERSION'
    self.gitcommit        = 'v' + self.version
    self.download         = ['https://github.com/nqomorita/monolis.git']
    self.liblist          = [['libmonolis.a']]
    self.includes         = ['monolis.h']
    #
    self.buildLanguages   = ['C','FC']
    self.downloadonWindows= 1
    self.hastests         = 0
    self.hastestsdatafiles= 0
    return

  def setupDependencies(self, framework):
    config.package.Package.setupDependencies(self, framework)
    self.mpi              = framework.require('config.packages.MPI',self)
    self.metis            = framework.require('config.packages.metis',self)
    self.openmp           = framework.require('config.packages.openmp',self)
    self.deps             = [self.mpi]
    self.odeps            = [self.metis,self.openmp]
    self.installdir      = framework.require('PETSc.options.installDir',self)
    return
    
  def configureLibrary(self):
    config.package.Package.configureLibrary(self)

  def Install(self):
    import os
    self.log.write('MonolisDir = '+self.packageDir+' installDir '+self.installDir+'\n')
    makewithargs=self.make.make+' lib FLAGS=INTEL,MPI,METIS '

    try:
      self.logPrintBox('Running configure on '+self.PACKAGE+'; this may take several minutes')
      output2,err2,ret2  = config.package.Package.executeShellCommand('cd '+self.packageDir + '  && '+'sh install_lib.sh'  +' && '+makewithargs, timeout=6000, log = self.log)
      libDir     = os.path.join(self.installDir, self.libdir)
      includeDir = os.path.join(self.installDir, self.includedir)
      self.logPrintBox('Running make on '+self.PACKAGE+'; this may take several minutes')
      output,err,ret = config.package.Package.executeShellCommandSeq(
          ['cp -f lib/libmonolis.a '+libDir+'/.',
           'cp -f include/*.* '+includeDir+'/.'
          ], cwd=self.packageDir, timeout=60, log = self.log)
    except RuntimeError as e:
      self.logPrint('Error running make on Monolis: '+str(e))
      raise RuntimeError('Error running make on ' + self.PACKAGE+': '+str(e))
    
    #self.logPrintBox('installDir:'+self.installDir+'   '+libDir+'   '+includeDir)
    self.addMakeRule('monolis-build','')
    self.addMakeRule('monolis-install','')
    return self.installDir

