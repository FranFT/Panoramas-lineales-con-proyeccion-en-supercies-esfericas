#include <iostream>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

/**********************************************
********** Declaraci�n anticipada *************
***********************************************/

//// Funciones auxiliares.
//Mat leeimagen(char* filename, int flagColor);
//void pintaI(const Mat& im, char* nombre_ventana = "imagen");
//void pintaI(const vector<Mat>& imagenes, char* nombre_ventana = "imagen");
//void pintaMI(const vector<Mat> &imagenes_solucion, char* nombre = "solucion");
//
//// Mapeado de coordenadas
//Mat mapeadoCilindrico(const Mat& imagen, float f, float r);
//Mat mapeadoEsferico(const Mat& imagen, float f, float r);
//vector<Mat> mapeadoCilindrico(const vector<Mat>& imagenes);
//vector<Mat> mapeadoEsferico(const vector<Mat>& imagenes);
//
//// Funcionalidad
//void cargar_imagenes(vector<Mat>& imagenes, int color = 0);
//void prueba_de_mapeado(const Mat& imagen);
//vector<Mat> recortar(const vector<Mat>& imagenes);
//int calcular_traslacion_relativa(const Mat& imagen1, const Mat& imagen2, bool cilindrico = true);
//Mat crearPanorama(const vector<Mat>& imagenes, bool cilindrico = true);
//
//// M�todo de Burt-Adelson
//void crearMascara(Mat & mascara, int posMezcla);
//void construirPiramideLap(Mat & im, Mat & imUltimo, vector<Mat_<Vec3f> > & pirLapIm, int niveles);
//void construirPiramideGaus(Mat & mascara, vector<Mat_<Vec3f> > & pirGausMasc, vector<Mat_<Vec3f> > & pirLapIm, Mat & imUltimo, int niveles);
//void mezclarPiramides(vector<Mat_<Vec3f> > & pirLapIm1, vector<Mat_<Vec3f> > & pirLapIm2, vector<Mat_<Vec3f> > & pirLapResultado, vector<Mat_<Vec3f> > & pirGausMascara, Mat & im1Ultimo, Mat & im2Ultimo, Mat & resUltimo, int niveles);
//void reconstruirPiramideResultado(vector<Mat_<Vec3f> > & pirResultado, Mat & resUltimo, Mat & resultado, int niveles);
//Mat mezclaImagenes(Mat & im1, Mat & im2);


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

// Muesta una imagen por pantalla en una ventana de nombre 'nombre_ventana'.
void pintaI(const vector<Mat>& imagenes, char* nombre_ventana = "imagen"){
	vector<Mat>::const_iterator it;
	for (it = imagenes.begin(); it != imagenes.end(); ++it){
		Mat im_copia = Mat(*it);
		im_copia.convertTo(im_copia, CV_8U);

		namedWindow(nombre_ventana, 1);
		imshow(nombre_ventana, im_copia);
		waitKey();
		destroyWindow(nombre_ventana);
	}
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


/**********************************************
*********** M�todo de Burt-Adelson ************
***********************************************/

// Crea la mascara necesaria para fundir las dos im�genes
void crearMascara(Mat & mascara, int posMezcla){

	// Rellena la mascara de 1s hasta donde se mezclan las dos im�genes
	mascara(Range::all(), Range(0, posMezcla)) = 1.0;

	// Calcular los valores de la regi�n para el suavizado de la fusi�n
	int anchoRegion = 50;
	float valorRegion = (1.f / anchoRegion);
	cout << valorRegion << endl;
	int j = posMezcla - (anchoRegion / 2);
	for (int i = 1; i < anchoRegion + 1; i++){
		mascara(Range::all(), Range(j, j + 1)) = 1.0 - (valorRegion*i);
		j++;
	}
}

// Funci�n que crea la pir�mide Laplaciana de una imagen
void construirPiramideLap(Mat & im, Mat & imUltimo, vector<Mat_<Vec3f> > & pirLapIm, int niveles){

	Mat temp1, temp2, temp3;
	Mat Lap;

	// Generar la pir�mide Laplaciana de la imagen
	im.copyTo(temp1);
	for (int i = 0; i < niveles; i++){
		pyrDown(temp1, temp2);
		pyrUp(temp2, temp3, temp1.size());
		Lap = temp1 - temp3;
		pirLapIm.push_back(Lap);
		temp2.copyTo(temp1);
	}

	// Se guarda la transformaci�n Gaussiana del �ltimo nivel de la pir�mide
	temp2.copyTo(imUltimo);
}

// Funci�n que crea la pir�mide Gaussiana de la m�scara
void construirPiramideGaus(Mat & mascara, vector<Mat_<Vec3f> > & pirGausMasc, vector<Mat_<Vec3f> > & pirLapIm, Mat & imUltimo, int niveles){

	Mat actual;
	// Generar la pir�mide Laplaciana de la m�scara
	cvtColor(mascara, actual, CV_GRAY2BGR);
	pirGausMasc.push_back(actual);
	actual = mascara;
	for (int i = 1; i < niveles + 1; i++){
		Mat down;

		if (pirLapIm.size() > i) {
			pyrDown(actual, down, pirLapIm[i].size());
		}
		else {
			pyrDown(actual, down, imUltimo.size());
		}

		Mat temp;
		cvtColor(down, temp, CV_GRAY2BGR);
		pirGausMasc.push_back(temp);
		down.copyTo(actual);
	}

}

// Funci�n que mezcla las dos pir�mides Laplacianas con la Gaussiana
void mezclarPiramides(vector<Mat_<Vec3f> > & pirLapIm1, vector<Mat_<Vec3f> > & pirLapIm2, vector<Mat_<Vec3f> > & pirLapResultado, vector<Mat_<Vec3f> > & pirGausMascara, Mat & im1Ultimo, Mat & im2Ultimo, Mat & resUltimo, int niveles){

	// Se mezclan los �ltimos niveles Gaussianos guardados despu�s de crear la pir�mide Laplaciana
	Mat mezclaNivel;
	resUltimo = (im1Ultimo.mul(pirGausMascara.back())) + (im2Ultimo.mul(Scalar(1.0, 1.0, 1.0) - pirGausMascara.back()));

	// Se mezclan el resto de niveles de cada una de las pir�mides
	for (int i = 0; i < niveles; i++){
		mezclaNivel = (pirLapIm1.at(i).mul(pirGausMascara.at(i))) + (pirLapIm2.at(i).mul(Scalar(1.0, 1.0, 1.0) - pirGausMascara.at(i)));
		pirLapResultado.push_back(mezclaNivel);
	}

}

// Reconstrucci�n de la imagen resultado con la pir�mide
void reconstruirPiramideResultado(vector<Mat_<Vec3f> > & pirResultado, Mat & resUltimo, Mat & resultado, int niveles){

	// Se reconstruye la imagen resultante de la fusi�n de pir�mides
	Mat temp1 = resUltimo;
	for (int i = niveles - 1; i >= 0; i--){
		Mat up;
		pyrUp(temp1, up, pirResultado.at(i).size());
		temp1 = up + pirResultado.at(i);

	}
	temp1.copyTo(resultado);
}

// Funci�n general que fusiona dos im�genes
Mat mezclaImagenes(Mat & im1, Mat & im2, int desplazamiento, int ancho){

	// Variables de almacenamiento
	int niveles = 3;
	vector<Mat_<Vec3f> > piramideLaplacianIm1, piramideLaplacianIm2;
	vector<Mat_<Vec3f> > piramideGaussianaMascara;
	vector<Mat_<Vec3f> > piramideResultado;
	Mat im1Ultimo, im2Ultimo, resUltimo;
	Mat roi;

	// Conversi�n del tipo de las im�genes de entrada a escala de grises
	Mat_<Vec3f> im1Aux; im1.convertTo(im1Aux, CV_32F, 1.0 / 255.0);
	Mat_<Vec3f> im2Aux; im2.convertTo(im2Aux, CV_32F, 1.0 / 255.0);
	Mat_<Vec3f> imagenResultado;

	// Creaci�n de la m�scara con la difusi�n colocada en la uni�n de ambas im�genes
	Mat_<float> mascara(im1Aux.rows, ancho, 0.0);
	//Mat_<float> mascara(im1Aux.rows, im1Aux.cols, 0.0);
	//int desplazamiento = mascara.cols / 2;
	crearMascara(mascara, desplazamiento);

	Mat_<Vec3f> _im1Aux = Mat::zeros(im1Aux.rows, ancho, im1Aux.type());
	roi = Mat(_im1Aux, Rect(Point2i(0, 0), im1Aux.size()));
	im1Aux.copyTo(roi);


	Mat_<Vec3f> _im2Aux = Mat::zeros(im2Aux.rows, ancho, im2Aux.type());
	roi = Mat(_im2Aux, Rect(Point2i(ancho - im2Aux.cols, 0), im2Aux.size()));
	im2Aux.copyTo(roi);

	
	// Construcci�n de las pir�mides Laplacianas de las im�genes y de la Gaussiana de la m�scara
	construirPiramideLap(_im1Aux, im1Ultimo, piramideLaplacianIm1, niveles);
	construirPiramideLap(_im2Aux, im2Ultimo, piramideLaplacianIm2, niveles);
	construirPiramideGaus(mascara, piramideGaussianaMascara, piramideLaplacianIm1, im1Ultimo, niveles);

	// Mezclado de las pir�mides Laplacianas con la Gaussiana
	mezclarPiramides(piramideLaplacianIm1, piramideLaplacianIm2, piramideResultado, piramideGaussianaMascara, im1Ultimo, im2Ultimo, resUltimo, niveles);

	// Reconstrucci�n de la pir�mide resultado
	reconstruirPiramideResultado(piramideResultado, resUltimo, imagenResultado, niveles);

	// Formateo de la imagen resultado
	Mat salida; imagenResultado.convertTo(salida, CV_8U, 255.0);
	return salida;
}


/**********************************************
********** Mapeado de coordenadas *************
***********************************************/
// http://math.etsu.edu/multicalc/prealpha/Chap3/Chap3-5/part1.htm
// http://stackoverflow.com/questions/12017790/warp-image-to-appear-in-cylindrical-projection

Mat mapeadoCilindrico(const Mat& imagen, float f, float r){
	// Variables necesarias.
	Mat canvas;
	float _x, _y;
	int cx = imagen.rows / 2;
	int cy = imagen.cols / 2;

	// Elegimos el tipo de representaci�n en funci�n de los canales de la imagen de entrada.
	if (imagen.channels() == 1)
		canvas = Mat(imagen.rows, imagen.cols, CV_8UC1);
	else if (imagen.channels() == 3)
		canvas = Mat(imagen.rows, imagen.cols, CV_8UC3);

	// Para cada uno de los p�xeles (x,y) de la imagen de entrada:
	// [1]: Hago un cambio en el sistema de referencia de coordenadas. Para ello traslado el
	//		origen de coordenadas hacia el centro de la imagen.
	// [2]: Aplico las ecuaciones para la proyecci�n cil�ndrica al rayo 3D (x,y,f) que pasa por el
	//		pixel (x,y) de la imagen.
	// [3]: Deshago el cambio del origen de coordenas a la vez que inserto el valor correspondiente
	//		de intensidad luminosa de la imagen original en las coordenadas resultantes de la proyecci�n cil�ndrica.
	for (int y = 0; y < imagen.cols; y++){
		_y = y - cy;										// [1]
		_y = r * atan(_y / f);								// [2]

		for (int x = 0; x < imagen.rows; x++){
			_x = x - cx;									// [1]
			_x = r * (_x / sqrt(pow(_y, 2) + pow(f, 2)));	// [2]		

			// [3]
			if (imagen.channels() == 1){
				if (_x + cx > 0 && _x < canvas.rows && _y + cy > 0 && _y < canvas.rows)
					canvas.at<uchar>(static_cast<int>(_x + cx), static_cast<int>(_y + cy)) = imagen.at<uchar>(x, y);
				else
					cout << "Warning: Pixel fuera de rango." << endl;
			}
			else if (imagen.channels() == 3){
				if (_x + cx > 0 && _x < canvas.rows && _y + cy > 0 && _y < canvas.rows){
					canvas.at<Vec3b>(static_cast<int>(_x + cx), static_cast<int>(_y + cy))[0] = imagen.at<Vec3b>(x, y)[0];
					canvas.at<Vec3b>(static_cast<int>(_x + cx), static_cast<int>(_y + cy))[1] = imagen.at<Vec3b>(x, y)[1];
					canvas.at<Vec3b>(static_cast<int>(_x + cx), static_cast<int>(_y + cy))[2] = imagen.at<Vec3b>(x, y)[2];
				}
				else
					cout << "Warning: Pixel fuera de rango." << endl;
			}
		}
	}

	// Devuelvo la imagen mapeada.
	return canvas;
}

Mat mapeadoEsferico(const Mat& imagen, float f, float r){
	// Variables necesarias.
	Mat canvas;
	float _x, _y;
	int cx = imagen.rows / 2;
	int cy = imagen.cols / 2;

	// Elegimos el tipo de representaci�n en funci�n de los canales de la imagen de entrada.
	if (imagen.channels() == 1)
		canvas = Mat(imagen.rows, imagen.cols, CV_8UC1);
	else if (imagen.channels() == 3)
		canvas = Mat(imagen.rows, imagen.cols, CV_8UC3);

	// Para cada uno de los p�xeles (x,y) de la imagen de entrada:
	// [1]: Hago un cambio en el sistema de referencia de coordenadas. Para ello traslado el
	//		origen de coordenadas hacia el centro de la imagen.
	// [2]: Aplico las ecuaciones para la proyecci�n cil�ndrica al rayo 3D (x,y,f) que pasa por el
	//		pixel (x,y) de la imagen.
	// [3]: Deshago el cambio del origen de coordenas a la vez que inserto el valor correspondiente
	//		de intensidad luminosa de la imagen original en las coordenadas resultantes de la proyecci�n cil�ndrica.
	for (int x = 0; x < imagen.rows; x++){
		_x = x - cx;											// [1]
		_x = r * atan(_x / f);									// [2]

		for (int y = 0; y < imagen.cols; y++){
			_y = y - cy;										// [1]
			_y = r * atan(_y / sqrt(pow(_x, 2) + pow(f, 2)));	// [2]

																// [3]
			if (imagen.channels() == 1){
				if (_x + cx > 0 && _x < canvas.rows && _y + cy > 0 && _y < canvas.rows)
					canvas.at<uchar>(static_cast<int>(_x + cx), static_cast<int>(_y + cy)) = imagen.at<uchar>(x, y);
				else
					cout << "Warning: Pixel fuera de rango." << endl;
			}
			else if (imagen.channels() == 3){
				if (_x + cx > 0 && _x < canvas.rows && _y + cy > 0 && _y < canvas.rows){
					canvas.at<Vec3b>(static_cast<int>(_x + cx), static_cast<int>(_y + cy))[0] = imagen.at<Vec3b>(x, y)[0];
					canvas.at<Vec3b>(static_cast<int>(_x + cx), static_cast<int>(_y + cy))[1] = imagen.at<Vec3b>(x, y)[1];
					canvas.at<Vec3b>(static_cast<int>(_x + cx), static_cast<int>(_y + cy))[2] = imagen.at<Vec3b>(x, y)[2];
				}
				else
					cout << "Warning: Pixel fuera de rango." << endl;
			}
		}
	}

	// Devuelvo la imagen mapeada.
	return canvas;
}

// Aplica el mapeado cil�ndrico a un conjunto de imagenes.
vector<Mat> mapeadoCilindrico(const vector<Mat>& imagenes){
	// Variables necesarias.
	vector<Mat> salida;
	vector<Mat>::const_iterator it;

	// Calculo la distacia focal estandar.
	float focal_lenght = imagenes.at(0).size().width / 2;

	// Aplico la proyecci�n a cada una de las im�genes.
	for (it = imagenes.begin(); it != imagenes.end(); ++it)
		salida.push_back(mapeadoCilindrico((*it), focal_lenght, focal_lenght));

	return salida;
}

// Aplica el mapeado esf�rico a un conjunto de imagenes.
vector<Mat> mapeadoEsferico(const vector<Mat>& imagenes){
	// Variables necesarias.
	vector<Mat> salida;
	vector<Mat>::const_iterator it;

	// Calculo la distacia focal estandar.
	float focal_lenght = imagenes.at(0).size().width / 2;

	// Aplico la proyecci�n a cada una de las im�genes.
	for (it = imagenes.begin(); it != imagenes.end(); ++it)
		salida.push_back(mapeadoEsferico((*it), focal_lenght, focal_lenght));

	return salida;
}

/**********************************************
**************** Funcionalidad ****************
***********************************************/
// Lee y almacena las im�genes en un vector.
void cargar_imagenes(vector<Mat>& imagenes, int color = 0){
	int num_imagenes = 5;
	char* nombres[] = { "imagenes/img1.jpg", "imagenes/img2.jpg", "imagenes/img3.jpg",
		"imagenes/img4.jpg", "imagenes/img5.jpg"};
	for (int i = 0; i < num_imagenes; i++){
		Mat temp = leeimagen(nombres[i], color);
		imagenes.push_back(temp);
	}
}
// Realiza la proyeccion cilindrica y esferica para distintos valores de f.
void prueba_de_mapeado(const Mat& imagen){
	vector<Mat> PCilindricas, PEsfericas;
	float focal_lenght;

	for (int i = 0; i < 3; i++){
		focal_lenght = 150 + 100*i;
		PCilindricas.push_back(mapeadoCilindrico(imagen, focal_lenght, focal_lenght));
		PEsfericas.push_back(mapeadoEsferico(imagen, focal_lenght, focal_lenght));
	}

	pintaMI(PCilindricas, "P.Cilindrica con distintos valores de F");
	pintaMI(PEsfericas, "P.Esferica con distintos valores de F");
}
// Ajusta los bordes laterales de la imagen.
vector<Mat> recortar(const vector<Mat>& imagenes){
	// Variables necesarias
	vector<Mat> salida;
	vector<Mat>::const_iterator it;
	int mitad;
	int topeIzquierda = -1;
	int topeDerecha = -1;

	// Ajustamos cada imagen al borde de la ventana.
	for (it = imagenes.begin(); it != imagenes.end(); ++it){
		// Creamos una copia para facilitar el acceso a los pixels en cualquier caso.
		Mat copia = (*it).clone();
		if (copia.channels() == 3)
			cvtColor(copia, copia, CV_RGB2GRAY);

		mitad = (*it).rows / 2;

		// Buscamos el primer pixel con contenido por la izquierda y por la derecha en la fila central de la imagen.
		for (int i = 0; i < (*it).cols && topeIzquierda == -1; i++){
			if (copia.at<uchar>(mitad, i) != NULL)
				topeIzquierda = i;
		}

		for (int i = (*it).cols - 1; i > 0 && topeDerecha == -1; i--)
			if (copia.at<uchar>(Point2i(i, mitad)) != NULL)
				topeDerecha = i;

		// Copiamos la secci�n deseada.
		Mat recortada = Mat((*it),Rect(topeIzquierda,0,(*it).cols-(topeIzquierda+((*it).cols-topeDerecha)),(*it).rows));
		salida.push_back(recortada);
	}

	return salida;
}

int calcular_traslacion_relativa(const Mat& imagen1, const Mat& imagen2, bool cilindrico = true){
	// Variables necesarias.
	int ancho_region, alto_region, traslacion, desplazamiento_region;
	int num_pixels_traslacion = 0;
	bool dentro_imagen = true;
	double error, error_min;

	Mat _imagen1 = imagen1.clone();
	Mat _imagen2 = imagen2.clone();
	Point2i inicio_region_fija, inicio_region_movil;
	Size size_region;

	// Trato la entrada.
	if (_imagen1.channels() == 3 || _imagen2.channels() == 3){
		cvtColor(_imagen1, _imagen1, CV_RGB2GRAY);
		cvtColor(_imagen2, _imagen2, CV_RGB2GRAY);
	}

	// Se desea alinear la imagen1 con la imagen2 por la derecha de la imagen1:
	//		|imagen1| <--- |imagen2|
	if (cilindrico)
		desplazamiento_region = 0;
	else
		desplazamiento_region = _imagen1.size().width / 6;

	// Ajusto los par�metros de las regiones que ser�n comparadas en cada imagen.
	traslacion = 0;
	error = 0.0;
	error_min = 0.0;

	// Defino el tama�o de la region.
	ancho_region = 60;
	alto_region = _imagen1.size().height / 2;
	desplazamiento_region = 0;
	size_region = Size(ancho_region, alto_region);

	// Fijo la regi�n de referencia a la hora de comparar im�genes, que ser� la regi�n fija.
	inicio_region_fija = Point2i(desplazamiento_region, _imagen2.size().height / 4);
	Mat region_fija = Mat(_imagen2, Rect(inicio_region_fija, size_region));

	// Mientras la regi�n m�vil se encuentre dentro de la imagen.
	while (dentro_imagen){

		// Compruebo si con el siguiente movimiento estar� dentro de la imagen.
		if (_imagen1.cols - size_region.width - traslacion > 0)
			inicio_region_movil = Point2i(_imagen1.cols - size_region.width - traslacion, inicio_region_fija.y );
		else
			dentro_imagen = false;

		// Si se encuentra dentro:
		if (dentro_imagen){
			// Defino dicha regi�n.
			Mat region_movil = Mat(_imagen1, Rect(inicio_region_movil, size_region));

			// Calculo el error cuadr�tico de la regi�n fija y la regi�n m�vil.
			for (int x = 0; x < region_movil.cols; x++){
				for (int y = 0; y < region_movil.rows; y++){
					error += pow(region_fija.at<uchar>(Point2i(x, y)) - region_movil.at<uchar>(Point2i(x, y)), 2);
				}
			}

			// Si es el primer error calculado, o es menor que el error que ya hab�a, actualizamos el error m�nimo encontrado
			// y la traslaci�n necesaria para llegar a ese error.
			if (traslacion == 0 || error < error_min){
				error_min = error;
				num_pixels_traslacion = traslacion;
			}

			// Actualizamos par�metros para la siguiente iteraci�n.
			error = 0.0;
			traslacion++;
		}			
	}

	return num_pixels_traslacion + ancho_region + desplazamiento_region;
}

Mat crearPanorama(const vector<Mat>& imagenes, bool cilindrico = true){
	int ancho = 0;
	int tipo = imagenes.at(0).type();

	vector<int> traslaciones;
	vector<Mat> ProyeccionImagen, RecorteProyeccion;
	Mat panorama;
	Point2i posicion_de_copiado = Point2i(0, 0);
	Mat roi;

	// Realizamos la proyecci�n seleccionada de las im�genes.
	if (cilindrico)
		ProyeccionImagen = mapeadoCilindrico(imagenes);
	else
		ProyeccionImagen = mapeadoEsferico(imagenes);

	// Ajustamos los bordes derecho e izquierdo.
	RecorteProyeccion = recortar(ProyeccionImagen);

	// Calculo la traslaci�n necesaria para cada par de im�genes.
	for (int i = 0; i < imagenes.size()-1; i++)
		traslaciones.push_back(calcular_traslacion_relativa(RecorteProyeccion.at(i), RecorteProyeccion.at(i + 1)));

	// Calculo el ancho final del panorama.
	ancho = RecorteProyeccion.at(0).cols;
	for (int i = 0; i < traslaciones.size(); i++)
		ancho = ancho + (RecorteProyeccion.at(i + 1).cols - traslaciones.at(i));
		
	// Inicialmente el panorama contiene la primera imagen por la izquierda.
	panorama = Mat(RecorteProyeccion.at(0).rows, ancho, tipo);
	roi = Mat(panorama, Rect(posicion_de_copiado, RecorteProyeccion.at(0).size()));
	RecorteProyeccion.at(0).copyTo(roi);


	Mat mezclado = mezclaImagenes(RecorteProyeccion.at(1), RecorteProyeccion.at(2), traslaciones.at(1), RecorteProyeccion.at(1).cols + RecorteProyeccion.at(2).cols - traslaciones.at(1));
	pintaI(mezclado);

	// Vamos a�adiendo el resto de im�genes.
	for (int i = 0; i < traslaciones.size(); i++){
		posicion_de_copiado.x = posicion_de_copiado.x + (RecorteProyeccion.at(i + 1).cols - traslaciones.at(i));
		roi = Mat(panorama, Rect(posicion_de_copiado, RecorteProyeccion.at(i + 1).size()));
		RecorteProyeccion.at(i + 1).copyTo(roi);		
	}

	pintaI(panorama);
	return panorama;
}


int main(){
	// Variables necesarias.
	Mat tablero;
	vector<Mat> imagenes_mosaico;

	// Prueba del mapeado.
	//tablero = leeimagen("imagenes/Tablero.png", 0);
	//prueba_de_mapeado(tablero);

	/*
	*	Realizaci�n del mosaico.
	*/
	cargar_imagenes(imagenes_mosaico, 1);
	crearPanorama(imagenes_mosaico, true);
	crearPanorama(imagenes_mosaico,false);
}
