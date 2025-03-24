#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "LR2.h"
#include <image_draw_objects.h>
#include <QScrollBar>
#include <filesystem>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    codec = new image_codec();

    connect(ui->accept_filename, SIGNAL(clicked()), this, SLOT(acceptFilenameClicked()));
    connect(ui->button_render, SIGNAL(clicked()), this, SLOT(buttonRenderClicked()));
    connect(ui->button_save, SIGNAL(clicked()), this, SLOT(buttonSaveClicked()));
}

void MainWindow::updateImage()
{
    if (curr_image == nullptr)
        return;

    QGraphicsScene *scene = new QGraphicsScene;

    scene->addPixmap(QPixmap::fromImage(*curr_image).scaled(ui->graphicsView_image->width(), ui->graphicsView_image->height(), Qt::KeepAspectRatio));
    ui->graphicsView_image->setScene(scene);
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

    int scale = ui->textinp_scale->value();
    log("scale: " + QString::number(scale));

    int offset = ui->textinp_offset->value();
    log("offset: " + QString::number(offset));

    matrix_gray matrix(width, height);
    matrix.fill(255);

    log("");

    QString renderType = ui->comboBox_renderType->currentText();

    log("render type: " + renderType);
    log("starting rendering...");

    if (png_buffer == nullptr)
    {
        png_buffer = new std::vector<unsigned char>();
    }

    if (renderType == "polygons")
    {
        draw_polygons_filled(&matrix, curr_vertices, curr_polygons, scale, offset);
    }
    else if (renderType == "vertices")
    {
        draw_vertices(&matrix, curr_vertices, (unsigned char)0, scale, offset);
    }
    else
    {
        log("unsupported render type");
        return;
    }

    log("finished rendering.");
    log("");

    log("");
    log("encoding...");
    codec->encode(png_buffer, &matrix, ImageColorScheme::IMAGE_GRAY, 8);
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
