#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "obj_parser.h"
#include "image_draw_objects.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    void updateImage();
    ~MainWindow();

public slots:
    void acceptFilenameClicked();
    void buttonRenderClicked();
    void buttonSaveClicked();
    void renderParamsChanged();
    void lockScale();
    void syncLockedScales();
    void chooseBgColorClicked();
    void chooseModelColorClicked();

private:
    scene* curr_scene;

    QImage* curr_image = nullptr;
    std::string image_basename = "";
    std::vector<unsigned char>* png_buffer = nullptr;

    model_renderer* curr_model = nullptr;
    unsigned char curr_bgColor[3] = {255, 255, 255};
    unsigned char curr_modelColor[3] = {255, 255, 255};

    void log(const QString txt);
    Ui::MainWindow *ui;

protected:
    void resizeEvent(QResizeEvent *event);
};
#endif // MAINWINDOW_H
