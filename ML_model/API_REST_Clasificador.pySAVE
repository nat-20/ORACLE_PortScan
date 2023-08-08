from bottle import Bottle, run, request
import json
import numpy as np
from sklearn.externals import joblib


app = Bottle()
aciertosT = 0;
total = 0;

@app.route('/clasificar', method='POST')
def alarma():
	global clf
	global normalizador
	global mean
	global std
	global aciertosT
	global total

	r = json.load(request.body) # se extrae el body de la peticion POST
	#print(r, len(r))
	r = np.matrix(r, dtype=np.float64)
	features = r[:,0:4]
	tags = np.array(r[:,4].T,dtype=int)

	features_N = (features-mean)/(std) # se normalizan las features
	y = clf.predict(features_N) # clasificador

	trues = 0
	tam = len(y)
	for i in range(0,tam):
		if y[i] == tags[0,i]:
			trues += 1

	# se adicional a los aciertos totales	
	aciertosT += trues
	total += tam 

	print("Classificación: ",y)
	print("Reales: ", tags)
	print("aciertos: ", aciertosT, "desaciertos: ",total - aciertosT, "accuracy: ",aciertosT/total)
	print("--------------------------------------------------")


def run():
    try:
        app.run(host='0.0.0.0', port=5000)
    except:
        print("An exception occurred")



if __name__ == "__main__":

	#------------------------ Initializar modelo ML --------------------------------
	# cargar la matriz con las medias y desviacion estandar del conjunto de entrenamiento
	#estandarizadores = joblib.load('SVM_stand.joblib') # del modelo svm
	
	estandarizadores = joblib.load('RF_Stand.joblib') # del modelo RF
	estandarizadores = np.concatenate((estandarizadores[0], estandarizadores[1]), axis=0)
	print(estandarizadores) 
	
	mean = estandarizadores[0,:]
	std = estandarizadores[1,:]

	#cargar el modelo de machine learning
	#clf = joblib.load('SVM_model.joblib') # del modelo svm
	clf = joblib.load('RF_model.joblib') # del modelo RF
	print("-------------------- Modelo ML -------------------")
	print(clf)
	print("--------------------------------------------------")
	#------------------------------------------------------------------------------

	run() #arranca el servicio API-REST
