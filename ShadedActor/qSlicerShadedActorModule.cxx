/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

// Qt includes
#include <QtPlugin>

// ShadedActor Logic includes
#include <vtkSlicerShadedActorLogic.h>

// ShadedActor includes
#include "qSlicerShadedActorModule.h"
#include "qSlicerShadedActorModuleWidget.h"

//-----------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(qSlicerShadedActorModule, qSlicerShadedActorModule);

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerShadedActorModulePrivate
{
public:
  qSlicerShadedActorModulePrivate();
};

//-----------------------------------------------------------------------------
// qSlicerShadedActorModulePrivate methods

//-----------------------------------------------------------------------------
qSlicerShadedActorModulePrivate::qSlicerShadedActorModulePrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerShadedActorModule methods

//-----------------------------------------------------------------------------
qSlicerShadedActorModule::qSlicerShadedActorModule(QObject* _parent)
  : Superclass(_parent)
  , d_ptr(new qSlicerShadedActorModulePrivate)
{
}

//-----------------------------------------------------------------------------
qSlicerShadedActorModule::~qSlicerShadedActorModule()
{
}

//-----------------------------------------------------------------------------
QString qSlicerShadedActorModule::helpText() const
{
  return "This is a loadable module that can be bundled in an extension";
}

//-----------------------------------------------------------------------------
QString qSlicerShadedActorModule::acknowledgementText() const
{
  return "This work was partially funded by NIH grant NXNNXXNNNNNN-NNXN";
}

//-----------------------------------------------------------------------------
QStringList qSlicerShadedActorModule::contributors() const
{
  QStringList moduleContributors;
  moduleContributors << QString("John Doe (AnyWare Corp.)");
  return moduleContributors;
}

//-----------------------------------------------------------------------------
QIcon qSlicerShadedActorModule::icon() const
{
  return QIcon(":/Icons/ShadedActor.png");
}

//-----------------------------------------------------------------------------
QStringList qSlicerShadedActorModule::categories() const
{
  return QStringList() << "Examples";
}

//-----------------------------------------------------------------------------
QStringList qSlicerShadedActorModule::dependencies() const
{
  return QStringList();
}

//-----------------------------------------------------------------------------
void qSlicerShadedActorModule::setup()
{
  this->Superclass::setup();
}

//-----------------------------------------------------------------------------
qSlicerAbstractModuleRepresentation* qSlicerShadedActorModule
::createWidgetRepresentation()
{
  return new qSlicerShadedActorModuleWidget;
}

//-----------------------------------------------------------------------------
vtkMRMLAbstractLogic* qSlicerShadedActorModule::createLogic()
{
  return vtkSlicerShadedActorLogic::New();
}
