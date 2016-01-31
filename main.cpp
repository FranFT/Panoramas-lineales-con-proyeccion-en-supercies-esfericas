#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

/**********************************************
************ Variables globales ***************
***********************************************/
#define FOCAL_LENGHT 3.097


/**********************************************
*********** Funciones auxiliares **************
***********************************************/
//// Entrada / Salida
// Lee la imagen 'filename' en color o escala de grises en funci�n del par�metro 'flagColor'.
Mat leeimagen(char* filename, int flagColor){
	return imread(filename, flagColor);
}

// Muesta una imagen por pantalla en una ventana de nombre 'nombre_ventana'.
void pintaI(const Mat& im, char* nombre_ventana = "imagen"){
	Mat im_copia = Mat(im);
	im_copia.convertTo(im_copia, CV_8U);

	namedWindow(nombre_ventana, 1);
	imshow(nombre_ventana, im_copia);
	waitKey();
	destroyWindow(nombre_ventana);
}

// Muestra un vector de im�genes en una sola imagen llamada 'solucion' por defecto.
void pintaMI(const vector<Mat> &imagenes_solucion, char* nombre = "solucion"){
	// Variables necesarias.
	int tam_ventana_rows = 0;
	int tam_ventana_cols = 0;
	int x_inicio = 0;
	bool color = false;
	Mat solucion, roi, aux;
	vector<Mat>::const_iterator it;

	// Calculo el valor de la imagen que contrendr� el conjunto de im�genes soluci�n.
	for (it = imagenes_solucion.begin(); it != imagenes_solucion.end(); it++){
		tam_ventana_cols = tam_ventana_cols + (*it).cols;
		if (tam_ventana_rows < (*it).rows)
			tam_ventana_rows = (*it).rows;

		// Indico si hay im�genes en color.
		if ((*it).channels() == 3)
			color = true;
	}

	if (color)
		solucion = Mat(tam_ventana_rows, tam_ventana_cols, CV_8UC3);
	else
		solucion = Mat(tam_ventana_rows, tam_ventana_cols, CV_8UC1);

	for (it = imagenes_solucion.begin(); it != imagenes_solucion.end(); it++){

		// Defino d�nde se situar� la imagen en la matriz soluci�n.
		roi = Mat(solucion, Rect(x_inicio, 0, (*it).cols, (*it).rows));

		// Si la matriz soluci�n est� definida para im�genes en color y encontramos
		// una imagen en escala de grises, la convertimos.
		if (color == true && (*it).channels() == 1){
			aux = (*it).clone();
			aux.convertTo(aux, CV_8U);
			cvtColor(aux, aux, COLOR_GRAY2RGB);
			aux.copyTo(roi);
		}
		else
		{
			aux = (*it).clone();
			aux.convertTo(aux, CV_8U);
			aux.copyTo(roi);
		}

		// Actualizo la posici�n de inicio de la regi�n de inter�s.
		x_inicio = x_inicio + aux.cols;
	}

	pintaI(solucion, nombre);
}

// Muestra una matriz de im�genes en una sola imagen llamada 'solucion' por defecto.
// En este caso se pasa un vector de vectores de im�genes. La funci�n pintar� en la misma
// fila las im�genes pertenecientes al mismo vector de imagenes. Por tanto, dibujar� tantas
// filas de im�genes como longitud tenga el vector de vectores.
// Es una funci�n equivalente a la que pinta m�ltiples im�genes en un solo vector, solo que se
// repite el proceso para cada uno de los elementos del vector de vectores. Por este motivo no
// est� comentado el c�digo.
void pintaMI(const vector<vector<Mat> > &imagenes_solucion, char* nombre = "solucion"){
	int tam_ventana_rows = 0;
	int tam_ventana_cols = 0;
	bool color = false;
	int x_inicio = 0;
	int aux_x = 0;
	int aux_y = 0;

	vector<vector<Mat> >::const_iterator it_filas;
	vector<Mat>::const_iterator it_columnas;
	vector<int>::const_iterator it;
	vector<int> y_inicio;
	Mat solucion, roi, aux;


	y_inicio.push_back(0);

	for (it_filas = imagenes_solucion.begin(); it_filas != imagenes_solucion.end(); ++it_filas){

		aux_x = 0;
		aux_y = 0;

		for (it_columnas = (*it_filas).begin(); it_columnas != (*it_filas).end(); ++it_columnas){
			aux_x = aux_x + (*it_columnas).cols;

			if (aux_y < (*it_columnas).rows)
				aux_y = (*it_columnas).rows;

			if ((*it_columnas).channels() == 3)
				color = true;
		}

		if (aux_x > tam_ventana_cols)
			tam_ventana_cols = aux_x;

		tam_ventana_rows += aux_y;
		y_inicio.push_back(aux_y);
	}


	if (color)
		solucion = Mat(tam_ventana_rows, tam_ventana_cols, CV_8UC3);
	else
		solucion = Mat(tam_ventana_rows, tam_ventana_cols, CV_8UC1);

	for (it_filas = imagenes_solucion.begin(), it = y_inicio.begin(); it_filas != imagenes_solucion.end(); ++it_filas, ++it){

		x_inicio = 0;

		for (it_columnas = (*it_filas).begin(); it_columnas != (*it_filas).end(); ++it_columnas){

			roi = Mat(solucion, Rect(x_inicio, (*it), (*it_columnas).cols, (*it_columnas).rows));

			if (color == true && (*it_columnas).channels() == 1){
				aux = (*it_columnas).clone();
				aux.convertTo(aux, CV_8U);
				cvtColor(aux, aux, COLOR_GRAY2RGB);
				aux.copyTo(roi);
			}
			else{
				aux = (*it_columnas).clone();
				aux.convertTo(aux, CV_8U);
				aux.copyTo(roi);
			}

			x_inicio = x_inicio + (*it_columnas).cols;
		}
	}

	pintaI(solucion, nombre);
}

/**********************************************
********* Modificaci�n de im�genes ************
***********************************************/
// Dibuja en una imagen, dado un punto, una cruz de color (RGB) azul por defecto.
void marcar_punto(Mat &img, Point2f pt, Scalar& color, int tam_pixel = 2){
	// Se comprueba que el punto tiene valores correctos.
	if (pt.x >= 0 && pt.y >= 0){
		// Dibuja la cruz en el punto pt.
		line(img, pt - Point2f(0.0, static_cast<float>(tam_pixel)), pt + Point2f(0.0, static_cast<float>(tam_pixel)), color);
		line(img, pt - Point2f(static_cast<float>(tam_pixel), 0.0), pt + Point2f(static_cast<float>(tam_pixel), 0.0), color);
		/*line(img, pt - Point2f(0, tam_pixel), pt + Point2f(0, tam_pixel), color);
		line(img, pt - Point2f(tam_pixel, 0), pt + Point2f(tam_pixel, 0), color);*/
	}
}

// Haciendo uso de 'marcar_punto' dibuja un conjunto de puntos en una imagen.
Mat marcar_imagen(const Mat& img, vector<Point2f> pts, Scalar& color, int tam_pixel = 2){
	vector<Point2f>::const_iterator it;
	Mat aux = Mat(img);

	// Si la imagen est� en escala de grises, la convertimos a color.
	if (aux.channels() == 1)
		cvtColor(aux, aux, COLOR_GRAY2RGB);

	// Pintamos cada punto.
	for (it = pts.begin(); it != pts.end(); it++)
		marcar_punto(aux, (*it), color, tam_pixel);

	return aux;
}


/**********************************************
********** Mapeado de coordenadas *************
***********************************************/
// http://math.etsu.edu/multicalc/prealpha/Chap3/Chap3-5/part1.htm
// http://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection
/*Mat mapeadoCilindrico(const Mat& imagen, float f, float s, int xc = 0, int yc = 0){
	Mat canvas = Mat(imagen.rows + xc, imagen.cols + yc, CV_32F);
	int _x, _y = 0;

	// Recorro la imagen. Para cada pixel (x,y,f):
	for (int x = 0; x < imagen.rows; x++){
		// Mapeo la coordenada x = x'
		_x = (s * atan(x / f) ) + xc;
		// Mapeo la coordenada y = y'
		for (int y = 0; y < imagen.cols; y++){
			_y = (s*(y / sqrt((x*x) + (f*f)))) + yc;

			// Si las coordenadas mapeadas (x', y') est�n dentro del canvas.
			if (_x < canvas.rows && _y < canvas.cols){
				// Pinto el contenido del pixel (x,y) original en las coordenadas mapeadas (x',y').
				canvas.at<float>(_x, _y) = static_cast<float>(imagen.at<uchar>(x, y));
			}
			else{
				cout << "EERRROOORRR" << endl;
			}
		}
	}
	pintaI(canvas);
	return canvas;

}*/

Mat mapeadoCilindrico(const Mat& imagen, float f, float r, int xc = 0, int yc = 0){
	Mat canvas = Mat(imagen.rows + xc, imagen.cols + yc, CV_8UC1);
	float denominador;
	float _x, _y;
	float ro, h;

	for (float x = 0; x < imagen.cols; x++){

		denominador = 1 / sqrt(pow(x, 2) + pow(f, 2));
		_x = x * denominador;
		ro = asin(_x);
		_x = r*ro + xc;

		for (float y = 0; y < imagen.rows; y++){
			_y = y * denominador;
			h = _y;			
			_y = r*h + yc;

			if (_x < canvas.cols && _y < canvas.rows)
				canvas.at<uchar>(_y, _x) = imagen.at<uchar>(y, x);
			else
				cout << "Error" << endl;
		}
	}
	pintaI(canvas);
	return canvas;
}

/*Mat mapeadoEsferico(const Mat& imagen, float f, float r){
	Mat canvas = Mat(imagen.rows, imagen.cols, CV_8U);
	float denominador;
	float _x, _y;

	for (int x = 0; x < imagen.cols; x++){
		denominador = sqrt(pow(x, 2) + pow(f, 2));
		_x = r*atan(x / f);

		for (int y = 0; y < imagen.rows; y++){
			_y = r*atan(y / denominador);
			if (_x < canvas.cols && _y < canvas.rows)
				canvas.at<uchar>(_y, _x) = imagen.at<uchar>(y, x);
			else
				cout << "Error" << endl;
		}
	}


	pintaI(canvas);
	return canvas;
}*/

Mat mapeadoEsferico(const Mat& imagen, float f, float r){
	Mat canvas = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	float denominador;
	float _x, _y;
	int cx = imagen.rows / 2;
	int cy = imagen.cols / 2;

	for (int x = 0; x < imagen.rows; x++){
		_x = x - cx;
		_x = r * atan(_x / f);

		for (int y = 0; y < imagen.cols; y++){
			_y = y - cy;
			_y = r * atan(_y / sqrt(pow(_x, 2) + pow(f, 2)));

			if (_x + cx > 0 && _x < canvas.rows && _y + cy > 0 && _y < canvas.rows)
				canvas.at<uchar>(static_cast<int>(_x + cx), static_cast<int>(_y + cy)) = imagen.at<uchar>(x,y);
			else
				cout << "Error" << endl;
		}
	}
	pintaI(canvas);
	return canvas;
}

int main(){
	vector<Mat> imagenes;
	Mat img = leeimagen("img1.jpg", 0);
	//Mat img = Mat(10, 10, CV_8UC1);
	//mapeadoCilindrico(img, 20, 20, 10, 10);
	//mapeadoCilindrico(img, img.size().width / 2, img.size().width / 2, 10, 10);
	mapeadoEsferico(img, 500, 500);

}