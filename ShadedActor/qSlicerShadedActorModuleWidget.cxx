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
#include <QDebug>

// SlicerQt includes
#include "qSlicerShadedActorModuleWidget.h"
#include "ui_qSlicerShadedActorModuleWidget.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerShadedActorModuleWidgetPrivate: public Ui_qSlicerShadedActorModuleWidget
{
public:
  qSlicerShadedActorModuleWidgetPrivate();
};

//-----------------------------------------------------------------------------
// qSlicerShadedActorModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicerShadedActorModuleWidgetPrivate::qSlicerShadedActorModuleWidgetPrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerShadedActorModuleWidget methods

//-----------------------------------------------------------------------------
qSlicerShadedActorModuleWidget::qSlicerShadedActorModuleWidget(QWidget* _parent)
  : Superclass( _parent )
  , d_ptr( new qSlicerShadedActorModuleWidgetPrivate )
{
}

//-----------------------------------------------------------------------------
qSlicerShadedActorModuleWidget::~qSlicerShadedActorModuleWidget()
{
}

//-----------------------------------------------------------------------------
void qSlicerShadedActorModuleWidget::setup()
{
  Q_D(qSlicerShadedActorModuleWidget);
  d->setupUi(this);
  this->Superclass::setup();
}
