/*==============================================================================

  Program: 3D Slicer

  Copyright (c) Kitware Inc.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
  and was partially funded by NIH grant 3P41RR013218-12S1

==============================================================================*/

// FooBar Widgets includes
#include "qSlicerShadedActorFooBarWidget.h"
#include "ui_qSlicerShadedActorFooBarWidget.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ShadedActor
class qSlicerShadedActorFooBarWidgetPrivate
  : public Ui_qSlicerShadedActorFooBarWidget
{
  Q_DECLARE_PUBLIC(qSlicerShadedActorFooBarWidget);
protected:
  qSlicerShadedActorFooBarWidget* const q_ptr;

public:
  qSlicerShadedActorFooBarWidgetPrivate(
    qSlicerShadedActorFooBarWidget& object);
  virtual void setupUi(qSlicerShadedActorFooBarWidget*);
};

// --------------------------------------------------------------------------
qSlicerShadedActorFooBarWidgetPrivate
::qSlicerShadedActorFooBarWidgetPrivate(
  qSlicerShadedActorFooBarWidget& object)
  : q_ptr(&object)
{
}

// --------------------------------------------------------------------------
void qSlicerShadedActorFooBarWidgetPrivate
::setupUi(qSlicerShadedActorFooBarWidget* widget)
{
  this->Ui_qSlicerShadedActorFooBarWidget::setupUi(widget);
}

//-----------------------------------------------------------------------------
// qSlicerShadedActorFooBarWidget methods

//-----------------------------------------------------------------------------
qSlicerShadedActorFooBarWidget
::qSlicerShadedActorFooBarWidget(QWidget* parentWidget)
  : Superclass( parentWidget )
  , d_ptr( new qSlicerShadedActorFooBarWidgetPrivate(*this) )
{
  Q_D(qSlicerShadedActorFooBarWidget);
  d->setupUi(this);
}

//-----------------------------------------------------------------------------
qSlicerShadedActorFooBarWidget
::~qSlicerShadedActorFooBarWidget()
{
}
