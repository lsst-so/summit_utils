# mypy: disable-error-code="name-defined"

config.doWrite = False  # this task writes separately, no need for this
config.doSaturation = True  # saturation very important for roundness measurement in qfm
config.brighterFatterMaxIter = 2  # Uncomment this to remove test warning
config.doDeferredCharge = False  # no calib for this yet
config.doBootstrap = True
config.doApplyGains = False
config.doSuspect = False
config.defaultSaturationSource = "CAMERAMODEL"
