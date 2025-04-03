#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "LR2.h"
#include "vertex_tools.h"
#include <image_draw_objects.h>
#include <QScrollBar>
#include <QColorDialog>
#include <filesystem>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    codec = new image_codec();

    connect(ui->spinBox_scaleX, SIGNAL(valueChanged(double)), this, SLOT(syncLockedScales()));
    connect(ui->checkBox_lockScale, SIGNAL(clicked(bool)), this, SLOT(lockScale()));

    connect(ui->accept_filename, SIGNAL(clicked()), this, SLOT(acceptFilenameClicked()));
    connect(ui->button_render, SIGNAL(clicked()), this, SLOT(buttonRenderClicked()));
    connect(ui->button_save, SIGNAL(clicked()), this, SLOT(buttonSaveClicked()));
    connect(ui->pushButton_bg_colorDialog, SIGNAL(clicked()), this, SLOT(chooseBgColorClicked()));

    connect(ui->spinBox_scaleX, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));
    connect(ui->spinBox_scaleY, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));

    connect(ui->spinBox_offset_x, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));
    connect(ui->spinBox_offset_y, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));
    connect(ui->spinBox_offset_z, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));

    connect(ui->spinBox_rotation_x, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));
    connect(ui->spinBox_rotation_y, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));
    connect(ui->spinBox_rotation_z, SIGNAL(valueChanged(double)), this, SLOT(renderParamsChanged()));
     
    connect(ui->textinp_width, SIGNAL(valueChanged(int)), this, SLOT(renderParamsChanged()));
    connect(ui->textinp_height, SIGNAL(valueChanged(int)), this, SLOT(renderParamsChanged()));
}

void MainWindow::chooseBgColorClicked()
{
    QColorDialog colorPickerDialog;

    QColor chosenColor = colorPickerDialog.getColor(QColor(curr_bgColor[0], curr_bgColor[1], curr_bgColor[2]));

    curr_bgColor[0] = chosenColor.red();
    curr_bgColor[1] = chosenColor.green();
    curr_bgColor[2] = chosenColor.blue();

    renderParamsChanged();
}

void MainWindow::syncLockedScales()
{
    if (ui->checkBox_lockScale->isChecked())
    {
        ui->spinBox_scaleY->setValue(ui->spinBox_scaleX->value());
    }
}

void MainWindow::lockScale()
{
    if (ui->checkBox_lockScale->isChecked())
    {
        ui->spinBox_scaleY->setEnabled(false);
        ui->spinBox_scaleY->setValue(ui->spinBox_scaleX->value());
    }
    else
    {
        ui->spinBox_scaleY->setEnabled(true);
    }
}

void MainWindow::renderParamsChanged()
{
    syncLockedScales();

    if (ui->checkBox_interactiveRender->isChecked())
    {
        this->buttonRenderClicked();
    }
}

void MainWindow::updateImage()
{
    if (curr_image == nullptr)
        return;

    QGraphicsScene *scene = new QGraphicsScene;

    scene->addPixmap(QPixmap::fromImage(*curr_image).scaled(ui->graphicsView_image->width(), ui->graphicsView_image->height(), Qt::KeepAspectRatio));
    ui->graphicsView_image->setScene(scene);
}

template <typename T>
void render_model(matrix_color<T>* matrix, QString renderType, std::vector<vertex>* vertices, std::vector<polygon>* polygons, double* offsets, double* angles, double scaleX, double scaleY)
{
    std::vector<vertex> transformed_vertices(vertices->size());
    transformVertices(transformed_vertices.data(), vertices->data(), vertices->size(), offsets, angles);

    if (renderType == "polygons")
    {
        draw_polygons_filled(matrix, &transformed_vertices, polygons, scaleX, scaleY);
    }
    else if (renderType == "vertices")
    {
        unsigned char color[3] = {0, 0, 0};
        draw_vertices(matrix, &transformed_vertices, matrix->c_arr_to_element(color), scaleX, scaleY);
    }
    else
    {
        //log("unsupported render type");
        return;
    }
}

void MainWindow::buttonRenderClicked()
{
    if(curr_vertices == nullptr || curr_polygons == nullptr)
    {
        log("data is not loaded!");
        return;
    }

    unsigned width = ui->textinp_width->value();
    unsigned height = ui->textinp_height->value();

    log("dimensions: " + QString::number(width) + "x" + QString::number(height));

    double scaleX = ui->spinBox_scaleX->value();
    log("scale X: " + QString::number(scaleX));

    double scaleY = ui->spinBox_scaleY->value();
    log("scale Y: " + QString::number(scaleY));

    QString renderType = ui->comboBox_renderType->currentText();

    log("");

    double offsets[3] = { ui->spinBox_offset_x->value(), ui->spinBox_offset_y->value(), ui->spinBox_offset_z->value() };
    double angles[3] = { ui->spinBox_rotation_x->value(), ui->spinBox_rotation_y->value(), ui->spinBox_rotation_z->value() };

    log("render type: " + renderType);
    log("starting rendering...");

    matrix* img_matrix;
    ImageColorScheme colorScheme;

    if (curr_bgColor[0] == curr_bgColor[1] && curr_bgColor[1] == curr_bgColor[2])
    {
        colorScheme = ImageColorScheme::IMAGE_GRAY;
        matrix_gray* img_gray = new matrix_gray(width, height);
        img_gray->fill(curr_bgColor[0]);
        img_matrix = img_gray;
    }
    else
    {
        colorScheme = ImageColorScheme::IMAGE_RGB;
        matrix_rgb* img_rgb = new matrix_rgb(width, height);
        img_rgb->fill(img_rgb->c_arr_to_element(curr_bgColor));
        img_matrix = img_rgb;
    }

    switch (colorScheme)
    {
        case IMAGE_GRAY:
            render_model((matrix_gray*)img_matrix, renderType, curr_vertices, curr_polygons, offsets, angles, scaleX, scaleY);
        case IMAGE_RGB:
            render_model((matrix_rgb*)img_matrix, renderType, curr_vertices, curr_polygons, offsets, angles, scaleX, scaleY);
        default:
            break;
    }

    log("finished rendering.");
    log("");

    log("");
    log("encoding...");

    if (png_buffer == nullptr)
    {
        png_buffer = new std::vector<unsigned char>();
    }

    codec->encode(png_buffer, img_matrix, colorScheme, 8);
    log("encoded.");
    log("");

    std::string img_format_str;

    switch (codec->native_format())
    {
        case PNG:
            img_format_str = "PNG";
            break;
        case JPEG:
            img_format_str = "JPG";
            break;
        default:
            log("unsupported image format, can't display");
            return;
    }

    curr_image = new QImage();
    curr_image->loadFromData(png_buffer->data(), png_buffer->size(), img_format_str.c_str());

    updateImage();

    ui->label_vertexCount->setNum((double)curr_vertices->size());
    ui->label_polygonCount->setNum((double)curr_polygons->size());
}

void MainWindow::acceptFilenameClicked()
{
    QString filename = ui->textinp_filename->text();
    log("accepted filename: " + filename);
    image_basename = std::filesystem::path(filename.toStdString()).stem();

    curr_vertices = new std::vector<vertex>();
    curr_polygons = new std::vector<polygon>();

    log("");
    log("starting reading...");
    readObj(filename.toStdString(), curr_vertices, curr_polygons);
    log("finished reading.");
    log("");
    log("vertices: " + QString::number(curr_vertices->size()));
    log("polygons: " + QString::number(curr_polygons->size()));
    log("");
}

void MainWindow::buttonSaveClicked()
{
    log("");
    if (image_basename == "" || png_buffer == nullptr)
    {
        log("nothing to save");
        log("");
        return;
    }

    std::string ext = "";

    switch (codec->native_format())
    {
        case JPEG:
            ext = ".jpeg";
            break;
        case PNG:
            ext = ".png";
            break;
        default:
            log("unsuported image format, saving without extension");
    }
    
    std::string filepath = "output/"+ image_basename;
    log(QString::fromStdString("saving to file: " + filepath));

    codec->save_image_file(png_buffer, filepath);
    log("saved.");

    log("");
}

void MainWindow::log(const QString txt)
{
    ui->label_log->setText(ui->label_log->text() + txt + "\n");
    ui->label_log->repaint();
    ui->scrollArea_log->verticalScrollBar()->setValue(ui->label_log->height());
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    updateImage();
}

MainWindow::~MainWindow()
{
    delete ui;
}
