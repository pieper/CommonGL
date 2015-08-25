import os
import unittest
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# ShaderComputation
#

class ShaderComputation(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "ShaderComputation" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an example of using the vtkOpenGLShaderComputation class to perform
    some cool computation.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# ShaderComputationWidget
#

class ShaderComputationWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
    self.outputSelector.selectNodeUponCreation = True
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = True
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.1
    self.imageThresholdSliderWidget.minimum = -100
    self.imageThresholdSliderWidget.maximum = 100
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for computing the output image. Voxels that have intensities lower than this value will set to zero.")
    parametersFormLayout.addRow("Image threshold", self.imageThresholdSliderWidget)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

    # Refresh Apply button state
    self.onSelect()

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = ShaderComputationLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    imageThreshold = self.imageThresholdSliderWidget.value
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), imageThreshold, enableScreenshotsFlag)

#
# ShaderComputationLogic
#

class ShaderComputationLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() == None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True

  def takeScreenshot(self,name,description,type=-1):
    # show the message even if not taking a screen shot
    slicer.util.delayDisplay('Take screenshot: '+description+'.\nResult is available in the Annotations module.', 3000)

    lm = slicer.app.layoutManager()
    # switch on the type to get the requested window
    widget = 0
    if type == slicer.qMRMLScreenShotDialog.FullLayout:
      # full layout
      widget = lm.viewport()
    elif type == slicer.qMRMLScreenShotDialog.ThreeD:
      # just the 3D window
      widget = lm.threeDWidget(0).threeDView()
    elif type == slicer.qMRMLScreenShotDialog.Red:
      # red slice window
      widget = lm.sliceWidget("Red")
    elif type == slicer.qMRMLScreenShotDialog.Yellow:
      # yellow slice window
      widget = lm.sliceWidget("Yellow")
    elif type == slicer.qMRMLScreenShotDialog.Green:
      # green slice window
      widget = lm.sliceWidget("Green")
    else:
      # default to using the full window
      widget = slicer.util.mainWindow()
      # reset the type so that the node is set correctly
      type = slicer.qMRMLScreenShotDialog.FullLayout

    # grab and convert to vtk image data
    qpixMap = qt.QPixmap().grabWidget(widget)
    qimage = qpixMap.toImage()
    imageData = vtk.vtkImageData()
    slicer.qMRMLUtils().qImageToVtkImageData(qimage,imageData)

    annotationLogic = slicer.modules.annotations.logic()
    annotationLogic.CreateSnapShot(name, description, type, 1, imageData)

  def run(self, inputVolume, outputVolume, imageThreshold, enableScreenshots=0):
    """
    Run the actual algorithm
    """

    if not self.isValidInputOutputData(inputVolume, outputVolume):
      slicer.util.errorDisplay('Input volume is the same as output volume. Choose a different output volume.')
      return False

    logging.info('Processing started')

    # Compute the thresholded output volume using the Threshold Scalar Volume CLI module
    cliParams = {'InputVolume': inputVolume.GetID(), 'OutputVolume': outputVolume.GetID(), 'ThresholdValue' : imageThreshold, 'ThresholdType' : 'Above'}
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True)

    # Capture screenshot
    if enableScreenshots:
      self.takeScreenshot('ShaderComputationTest-Start','MyScreenshot',-1)

    logging.info('Processing completed')

    return True


class ShaderComputationTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    # slicer.mrmlScene.Clear(0)
    pass

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    #self.test_ShaderComputation1()
    self.test_ShaderComputation2()

  def test_ShaderComputation1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    # self.delayDisplay("Starting the test", 100)

    mrHeadVolume = slicer.util.getNode('MRHead')
    if not mrHeadVolume:
      import SampleData
      sampleDataLogic = SampleData.SampleDataLogic()
      print("Getting MR Head Volume")
      mrHeadVolume = sampleDataLogic.downloadMRHead()

    resize = vtk.vtkImageResize()
    resize.SetInputDataObject(mrHeadVolume.GetImageData())
    resize.SetOutputDimensions(256,256,128)
    resize.Update()

    from vtkSlicerShadedActorModuleLogicPython import *

    shaderComputation=vtkOpenGLShaderComputation()

    shaderComputation.SetVertexShaderSource("""
      #version 120
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec4 interpolatedColor;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedColor = vec4(0.5) + vec4(vertexAttribute, 1.);
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """)

    shaderComputation.SetFragmentShaderSource("""
      #version 120
      varying vec4 interpolatedColor;
      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D volumeSampler;
      void main()
      {
      /*
        gl_FragColor = vec4 ( 1.0, 0.0, 0.0, 1.0 );
        gl_FragColor = interpolatedColor;
        vec4 volumeSample = texture3D(volumeSampler, interpolatedTextureCoordinate);
        volumeSample *= 100;
        gl_FragColor = mix( volumeSample, interpolatedColor, 0.5);
      */

        vec4 integratedRay = vec4(0.);
        for (int i = 0; i < 256; i++) {
          vec3 samplePoint = vec3(interpolatedTextureCoordinate.st, i/256.);
          vec4 volumeSample = texture3D(volumeSampler, samplePoint);
          integratedRay += volumeSample;
        }
        gl_FragColor = integratedRay;
      }
    """)
    shaderComputation.SetTextureImageData(resize.GetOutputDataObject(0))

    resultImage = vtk.vtkImageData()
    resultImage.SetDimensions(512, 512, 1)
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    shaderComputation.SetResultImageData(resultImage)

    shaderComputation.Compute()

    iv = vtk.vtkImageViewer()
    iv.SetColorLevel(128)
    iv.SetColorWindow(256)
    iv.SetInputData(resultImage)
    iv.Render()

    slicer.modules.ShaderComputationWidget.iv = iv

  def sampleVolumeParameters(self,volumeNode):
    """Calculate the dictionary of substitutions for the
    current state of the volume node in a form for
    substitution into the sampleVolume shader function
    TODO: this would probably be better as uniforms, but that
    requires doing a lot of parsing and data management in C++
    """

    volumeArray = slicer.util.array(volumeNode.GetID())

    rasToIJK = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(rasToIJK)

    transformNode = volumeNode.GetParentTransformNode()
    if transformNode:
      if transformNode.IsTransformToWorldLinear():
        rasToRAS = vtk.vtkMatrix4x4()
        transformNode.GetMatrixTransformToWorld(rasToRAS)
        rasToRAS.Invert()
        rasToRAS.Multiply4x4(rasToIJK, rasToRAS, rasToIJK)
      else:
        print('Cannot handle nonlinear transforms')

    # dimensions are number of pixels in (row, column, slice)
    # which maps to 0-1 space of S, T, P
    dimensions = volumeNode.GetImageData().GetDimensions()
    ijkToSTP = vtk.vtkMatrix4x4()
    ijkToSTP.Identity()
    for diagonal in range(3):
      ijkToSTP.SetElement(diagonal,diagonal, 1./dimensions[diagonal])
    ijkToSTP.Multiply4x4(ijkToSTP, rasToIJK, rasToIJK)

    parameters = {}
    rows = ('rasToS', 'rasToT', 'rasToP')
    for row in range(3):
      rowKey = rows[row]
      parameters[rowKey] = ""
      for col in range(4):
        element = rasToIJK.GetElement(row,col)
        parameters[rowKey] += "%f," % element
      parameters[rowKey] = parameters[rowKey][:-1] # clear trailing comma
    return parameters

  def rayCastVolumeParameters(self,volumeNode):
    """Calculate the dictionary of substitutions for the
    current state of the volume node and camera in a form for
    substitution into the rayCast shader function
    TODO: this would probably be better as uniforms, but that
    requires doing a lot of parsing and data management in C++
    """

    parameters = {}

    rasBounds = [0,]*6
    volumeNode.GetRASBounds(rasBounds)
    parameters['rasBoxMin'] = "%f, %f, %f" % (rasBounds[0], rasBounds[2], rasBounds[4])
    parameters['rasBoxMax'] = "%f, %f, %f" % (rasBounds[1], rasBounds[3], rasBounds[5])

    # get the camera parameters from default 3D window
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    renderWindow = threeDView.renderWindow()
    renderers = renderWindow.GetRenderers()
    renderer = renderers.GetItemAsObject(0)
    camera = renderer.GetActiveCamera()

    import numpy
    viewPosition = numpy.array(camera.GetPosition())
    focalPoint = numpy.array(camera.GetFocalPoint())
    viewDistance = numpy.linalg.norm(focalPoint - viewPosition)
    viewNormal = (focalPoint - viewPosition) / viewDistance
    viewUp = numpy.array(camera.GetViewUp())
    viewAngle = camera.GetViewAngle()
    viewRight = numpy.cross(viewNormal,viewUp)

    parameters['eyeRayOrigin'] = "%f, %f, %f" % (viewPosition[0], viewPosition[1], viewPosition[2])
    parameters['viewNormal'] = "%f, %f, %f" % (viewNormal[0], viewNormal[1], viewNormal[2])
    parameters['viewRight'] = "%f, %f, %f" % (viewRight[0], viewRight[1], viewRight[2])
    parameters['viewUp'] = "%f, %f, %f" % (viewUp[0], viewUp[1], viewUp[2])

    return parameters

  def test_ShaderComputation2(self, caller=None, event=None):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    if False:
      self.delayDisplay("Starting the test", 100)

    volumeToRender = slicer.util.getNode('MRHead')
    if not volumeToRender:
      import SampleData
      sampleDataLogic = SampleData.SampleDataLogic()
      print("Getting CTA Volume")
      volumeToRender = sampleDataLogic.downloadMRHead()

    if not hasattr(self,"shaderComputation") and hasattr(slicer.modules.ShaderComputationInstance, "test"):
      oldSelf = slicer.modules.ShaderComputationInstance.test
      oldSelf.renderWindow.RemoveObserver(oldSelf.renderTag)

    if not hasattr(self,"shaderComputation"):
      print('new shaderComputation')
      from vtkSlicerShadedActorModuleLogicPython import vtkOpenGLShaderComputation
      self.shaderComputation=vtkOpenGLShaderComputation()

    # TODO: these strings can move to a CommonGL spot once debugged
    headerSource = """
      #version 120
    """

    self.shaderComputation.SetVertexShaderSource("""
      %(header)s
      attribute vec3 vertexAttribute;
      attribute vec2 textureCoordinateAttribute;
      varying vec3 interpolatedTextureCoordinate;
      void main()
      {
        interpolatedTextureCoordinate = vec3(textureCoordinateAttribute, .5);
        gl_Position = vec4(vertexAttribute, 1.);
      }
    """ % {
      'header' : headerSource
    })


    intersectBoxSource = """
      bool intersectBox(const in vec3 rayOrigin, const in vec3 rayDirection,
                        const in vec3 boxMin, const in vec3 boxMax,
                        out float tNear, out float tFar)
        // intersect ray with a box
        // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
      {
          // compute intersection of ray with all six bbox planes
          vec3 invRay = vec3(1.) / rayDirection;
          vec3 tBot = invRay * (boxMin - rayOrigin);
          vec3 tTop = invRay * (boxMax - rayOrigin);

          // re-order intersections to find smallest and largest on each axis
          vec3 tMin = min(tTop, tBot);
          vec3 tMax = max(tTop, tBot);

          // find the largest tMin and the smallest tMax
          float largest_tMin = max(max(tMin.x, tMin.y), max(tMin.x, tMin.z));
          float smallest_tMax = min(min(tMax.x, tMax.y), min(tMax.x, tMax.z));

          tNear = largest_tMin;
          tFar = smallest_tMax;

          return smallest_tMax > largest_tMin;
      }
    """

    sampleVolumeSource = """
      void sampleVolume(in sampler3D volumeSampler, in vec3 samplePoint, in float gradientSize,
                        out float sample, out vec3 normal, out float gradientMagnitude)
      {
        // vectors to map RAS to stp
        vec4 rasToS =  vec4( %(rasToS)s );
        vec4 rasToT =  vec4( %(rasToT)s );
        vec4 rasToP =  vec4( %(rasToP)s );

        vec3 stpPoint;
        vec4 sampleCoordinate = vec4(samplePoint, 1.);
        stpPoint.x = dot(rasToS,sampleCoordinate);
        stpPoint.y = dot(rasToT,sampleCoordinate);
        stpPoint.z = dot(rasToP,sampleCoordinate);

        // read from 3D texture
        sample = texture3D(volumeSampler, stpPoint).r;

        // central difference sample gradient (N is -1)
        float s100 = texture3D(volumeSampler, stpPoint + vec3(gradientSize,0,0)).r;
        float sN00 = texture3D(volumeSampler, stpPoint - vec3(gradientSize,0,0)).r;
        float s010 = texture3D(volumeSampler, stpPoint + vec3(0,gradientSize,0)).r;
        float s0N0 = texture3D(volumeSampler, stpPoint - vec3(0,gradientSize,0)).r;
        float s001 = texture3D(volumeSampler, stpPoint + vec3(0,0,gradientSize)).r;
        float s00N = texture3D(volumeSampler, stpPoint - vec3(0,0,gradientSize)).r;

        vec3 gradient = vec3( 0.5f*(s100-sN00),
                              0.5f*(s010-s0N0),
                              0.5f*(s001-s00N));
        gradientMagnitude = length(gradient);
        normal = normalize(gradient);
      }
    """ % self.sampleVolumeParameters(volumeToRender)

    rayCastParameters = self.rayCastVolumeParameters(volumeToRender)
    rayCastParameters.update({
          'rayMaxSteps' : 5000,
          'rayStepSize' : 0.01,
    }) # TODO: auto calculate ray parameters
    rayCastSource = """
      // volume ray caster - starts from the front and collects color and opacity
      // contributions until fully saturated.
      // Sample coordinate is 0->1 texture space
      vec4 rayCast( in vec3 sampleCoordinate, in sampler3D volumeSampler )
      {
        // TODO aspect: float aspect = imageW / (1.0 * imageH);
        vec2 normalizedCoordinate = 2. * (sampleCoordinate.st -.5);

        // calculate eye ray in world space
        vec3 eyeRayOrigin = vec3(%(eyeRayOrigin)s);
        vec3 eyeRayDirection;

        // ||viewNormal + u * viewRight + v * viewUp||

        eyeRayDirection = normalize (                            vec3( %(viewNormal)s )
                                      + normalizedCoordinate.x * vec3( %(viewRight)s  )
                                      + normalizedCoordinate.y * vec3( %(viewUp)s     ) );


        vec3 pointLight = vec3(250., 250., 400.); // TODO

        // find intersection with box, possibly terminate early
        float tNear, tFar;
        vec3 rasBoxMin = vec3( %(rasBoxMin)s );
        vec3 rasBoxMax = vec3( %(rasBoxMax)s );
        bool hit = intersectBox( eyeRayOrigin, eyeRayDirection, rasBoxMin, rasBoxMax, tNear, tFar );
        if (!hit) {
          return vec4(0.,0.,.5,1.); // TODO: mid blue background for now
        }

        if (tNear < 0.) tNear = 0.;     // clamp to near plane

        // march along ray from front, accumulating color and opacity
        vec4 integratedPixel = vec4(0.);
        float gradientSize = .05;
        float tCurrent = tNear + gradientSize;
        float sample;
        int rayStep;
        for(rayStep = 0; rayStep < %(rayMaxSteps)d; rayStep++) {

          vec3 samplePoint = eyeRayOrigin + eyeRayDirection * tCurrent;

          // TODO: add a vector field proportional to rayStep, u, v
          /*
          vec4 vectorField;
          float blend = rayStep * 1000. / %(rayMaxSteps)d;
          vectorField.x = blend * sin(90. * u) + cos(90. * v);
          vectorField.y = blend * -cos(90. * u) + sin(90. * v);
          vectorField.z = 0.;
          //samplePoint += vectorField;
          */

          vec3 normal;
          float gradientMagnitude;
          sampleVolume(volumeSampler, samplePoint, gradientSize, sample, normal, gradientMagnitude);


          // Phong lighting
          // http://en.wikipedia.org/wiki/Phong_reflection_model
          vec3 Cdiffuse = vec3(1.,1.,0.);
          vec3 Cspecular = vec3(1.,1.,1.);
          float Kdiffuse = 1.;
          float Kspecular = .5;
          float Shininess = 5.;

          vec3 V = normalize(eyeRayOrigin - samplePoint);
          vec3 L = normalize(pointLight - samplePoint);
          vec3 R = normalize(2.*(dot(L,normal))*normal - L);
          vec3 phongColor = vec3(0.);
          phongColor = phongColor + Kdiffuse * dot(L,normal) * Cdiffuse;
          phongColor = phongColor + Kspecular * pow( dot(R,V), Shininess ) * Cspecular;

          vec4 color;
          color = vec4(phongColor, 1.);
          color.a = (1.*sample/100. + gradientMagnitude/.1) * %(rayStepSize)f*.01;
          color.a = (1.*sample/.0001 + gradientMagnitude/.1) * %(rayStepSize)f*.01;
          color.a = 1.;
          color.a = clamp( color.a, 0., 1. );

          // accumulate result
          vec4 newPixel;
          //float a = color.a * 1. /*density*/; // here w is alpha
          //newPixel = mix( integratedPixel, color, vec4(a) );
          //newPixel.a = integratedPixel.a + a;
          //integratedPixel = newPixel;

    sample *=200.;
          if (sample > .001) {
            //return vec4(1., 0., 0., 1.);
          } else {
            //return vec4(1., 1., 0., 1.);
          }
          color = vec4(sample, sample, sample, sample/1000.);
          newPixel.rgb = 1-color.a * integratedPixel.rgb + color.a * color.rgb;
          newPixel.a = integratedPixel.a + color.a;
          integratedPixel = newPixel;

          tCurrent += %(rayStepSize)f;
          if (
              tCurrent >= tFar - gradientSize // stepped out of the volume
                ||
              integratedPixel.a >= 1.  // pixel is saturated
          ) {
            break; // we can stop now
          }
        }
        return (integratedPixel);
      }
    """ % rayCastParameters

    self.shaderComputation.SetFragmentShaderSource("""
      %(header)s
      %(intersectBox)s
      %(sampleVolume)s
      %(rayCast)s

      varying vec3 interpolatedTextureCoordinate;
      uniform sampler3D volumeSampler;
      void main()
      {
        gl_FragColor = rayCast(interpolatedTextureCoordinate, volumeSampler);
      }
    """ % {
      'header' : headerSource,
      'intersectBox' : intersectBoxSource,
      'sampleVolume' : sampleVolumeSource,
      'rayCast' : rayCastSource,
    })

    if True:
      print(self.shaderComputation.GetFragmentShaderSource())
      fp = open('/tmp/shader.glsl','w')
      fp.write(self.shaderComputation.GetFragmentShaderSource())
      fp.close()

    self.shaderComputation.SetTextureImageData(volumeToRender.GetImageData())

    resultImage = vtk.vtkImageData()
    resultImage.SetDimensions(512, 512, 1)
    resultImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 4)
    self.shaderComputation.SetResultImageData(resultImage)

    self.shaderComputation.Compute()

    # TODO:
    """
    - refactor test to support re-use of self.shaderComputation
    - fix vtkOpenGLShaderComputation to set MTime of resultImage (point data and image data)
    - test render performance
    - debug raycaster
    - add features!
    """

    if not hasattr(self, "iv"):
      self.iv = vtk.vtkImageViewer()
      self.iv.SetColorLevel(128)
      self.iv.SetColorWindow(256)
    self.iv.SetInputData(resultImage)
    self.iv.Render()

    if not hasattr(self,"renderWindow"):
      layoutManager = slicer.app.layoutManager()
      threeDWidget = layoutManager.threeDWidget(0)
      threeDView = threeDWidget.threeDView()
      self.renderWindow = threeDView.renderWindow()
      print('adding render observer')
      self.renderTag = self.renderWindow.AddObserver(vtk.vtkCommand.EndEvent, self.test_ShaderComputation2)
    slicer.modules.ShaderComputationInstance.test = self
