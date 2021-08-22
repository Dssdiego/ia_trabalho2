import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import io
import os
import zipfile

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from sklearn.utils import Bunch
from PIL import Image



# Converte os pixels da imagem para preto e/ou branco,
# 	de acordo com um threshold informado
def binarize_pixel(image_matrix, thresh):
    white = 255
    black = 0

    initial_conv = np.where((image_matrix <= thresh), image_matrix, white)
    final_conv = np.where((initial_conv > thresh), initial_conv, black)

    return final_conv


# Reduz o tamanho de uma imagem (64x64 >> 8x8)
# realizando a média entre blocos de 8x8
def reduce_blocks(image_src):
    return block_reduce(image_src/255/64, block_size=(8, 8)).round(2)


# Converte a imagem para preto e branco
def binarize(image_src):
    return binarize_pixel(image_matrix=image_src, thresh=128)


# Carrega as imagens da pasta assets/images
def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    filenames.sort()
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename))
        image_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(image_src)
    return images


# Faz as transformações necessárias nas imagens para melhor
# atender ao treinamento do modelo
def transform_images(images):
    # Convertendo imagens para preto e branco
    images_b = []
    for image in images:
        images_b.append(binarize(image))
    st.write('Originais (64x64)')
    st.info('Carregamos as imagens originais, em tons de cinza. Aqui cada imagem possui 64x64 = 4096 pixels')
    st.image(images)
    st.write('Convertidas (64x64)')
    st.info('Convertemos as imagens para preto e branco, facilitando o treino do modelo')
    st.image(images_b)

    # Reduz os blocos das imagens (64x64 >> 8x8)
    images_r = []
    for image_b in images_b:
        images_r.append(reduce_blocks(image_b))

    st.write('Reduzidas (8x8)')
    st.info('Reduzimos as imagens de 64x64 para 8x8, aplicando um algoritmo de média em blocos de 8x8.')
    st.warning('Este processo é necessário devido à grande quantidade de pixels que teríamos ao usar as imagens originais.\n\nComo cada imagem tem 64x64 = 4096 pixels e temos um total de 120 imagens, ao usar as imagens originais teríamos que treinar 491520 pixels.\n\nJá que reduzimos a quantidade de pixels nas imagens, teremos que trabalhar *somente* com 7680 pixels 😉')
    st.image(images_r)

    return images_r


# Faz o mapeamento de uma letra para um índice
# A = 0
# B = 1
# C = 2
# D = 3
# E = 4
def map_letter(letter):
    return ord(letter.lower()) - 97


# Faz o mapeamento de um índice para uma letra
# 0 = A
# 1 = B
# 2 = C
# 3 = D
# 4 = E
def map_number(number):
    return chr(ord('A')+number).upper()


# Prevê a letra conforme a imagem informada
# (usa o modelo treinado anteriormente)
def test_model(model, image_src):
    # TODO: Implementar

    st.image(image_src)

    image_b = binarize(image_src)
    st.image(image_b)

    st.write(image_b)
    image_r = reduce_blocks(image_b)
    st.image(image_r)

    # image_res = image_b.reshape((1, -1))

    # st.write(model.predict(image_res))

    # TODO: Se passou tudo 100%, mostra uma mensagem de sucesso
    # if img_pred is not None:
    #     st.success('Esta imagem é a letra ' + predict_letter(img_pred))

    
    return


# Treina o modelo (usando SVM)
# TODO: Deve ser feito para ANN (Artificial Neural Networks) também 
def train_model(bunch):

    st.markdown('*Amostra*')
    st.info('Essa é uma amostra dos dados usados pelo modelo. Cada imagem possui tamanho 8x8 e pixels que estão no intervalo 0-1, onde 0 é **preto** e 1 é **branco**')
    fig_before, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    for ax, image, label in zip(axes, bunch.images, bunch.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Letra: {map_number(label)}')

    st.pyplot(fig_before)

    # Faz um "flat" no array de imagens. Ou seja, transforma o array para ficar em uma única linha
    n_samples = len(bunch.images)
    data = bunch.images.reshape((n_samples, -1))

    st.write('Nivelamento dos Dados')
    st.info('Para aplicar um classificador nesses dados, precisamos nivelar as imagens, transformando cada array 2D de valores de tons de cinza de forma (8, 8) em forma (64,). Ou seja: Cada linha é uma imagem e cada coluna são seus pixels, descritos em um array de uma única dimensão (64)')
    st.write(data)

    # Divide os dados em 2 datasets: 50% dos dados para treino e 50% para testes
    X_train, X_test, y_train, y_test = train_test_split(data, bunch.target, test_size=0.5, shuffle=True)

    st.write('Divisão em conjuntos de treino e teste')
    st.info('Dividimos os dados na forma 50/50, onde metade é o conjunto de treino e a outra metade o conjunto de testes')
    st.write('Tamanho do conjunto de treino:', len(X_train))
    st.write('Tamanho do conjunto de testes:', len(X_test))

    # st.write('X_train', X_train)
    # st.write('y_train', y_train)
    # st.write('X_test', X_test)
    # st.write('y_test', y_test)

    ###########
    ### SVM ###
    ###########

    st.header("Modelo SVM")
    st.info('Treinamos os dados com o modelo *C-Support Vector Classification (SVC)* - Modelo de Support Vector Machines, prevemos e conferimos se os dados previstos "batem" com o conjunto de testes')

    # Treinamos o modelo SVM
    model = SVC(random_state=42)
    model.fit(X_train, y_train)

    # Testamos o modelo prevendo o conjunto de testes
    pred = model.predict(X_test)
    # st.write('Previsto (SVM)', pred)
    
    st.markdown('*5 Previstas (Amostra)*')
    fig_after, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, pred):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Previsto: {map_number(prediction)}')

    st.pyplot(fig_after)

    # st.write(classification_report(y_test, pred))
    acc = metrics.accuracy_score(y_test, pred).round(2)
    st.write("Acurácia:", acc)

    # Salvamos as informacoes do Modelo SVM
    svm_model = {}
    svm_model['model'] = model
    svm_model['pred'] = pred
    svm_model['y_test'] = y_test
    svm_model['score'] = acc

    ###########
    ### RNA ###
    ###########

    st.header("Modelo RNA")
    st.info('Treinamos os dados com o modelo *Multi-layer Perceptron classifier (MLPClassifier)* - Modelo de Rede Neural Artificial, prevemos e conferimos se os dados previstos "batem" com o conjunto de testes')

    model = MLPClassifier(hidden_layer_sizes=(64, 64), random_state=42)
    model.fit(X_train, y_train)

    # Testamos o modelo prevendo o conjunto de testes
    pred = model.predict(X_test)
    # st.write('Previsto (RNA)', pred)

    st.markdown('*5 Previstas (Amostra)*')
    fig_after, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, pred):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f'Previsto: {map_number(prediction)}')

    st.pyplot(fig_after)

    # st.write(classification_report(y_test, pred))
    acc = metrics.accuracy_score(y_test, pred).round(2)
    st.write("Acurácia:", acc)

    # Salvamos as informacoes do Modelo RNA
    rna_model = {}
    rna_model['model'] = model
    rna_model['pred'] = pred
    rna_model['y_test'] = y_test
    rna_model['score'] = acc

    return svm_model, rna_model


# Método principal
def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title('Trabalho 2 - Inteligência Artificial')
    st.markdown('**Aluno:** Diego Santos Seabra')
    st.markdown('**Matrícula:** 0040251')
    
    st.header('Processamento de Imagens')

    st.info('As imagens devem estar disponíveis na pasta "assets/images" no formato ".png" e tamanho 64x64')

    # Lê todos os arquivos de imagem dentro da pasta asseto
    images = load_images_from_folder('./assets/images')
    # st.image(images)

    images_r = transform_images(images)
    
    # Leitura do arquivo de mapeamento (imagens > letras)
    st.header('Tabela de mapeamento')
    st.info('Esta tabela mapeia os arquivos de imagem para seus devidos rótulos\n\nimage_file: nome do arquivo (sem o formato)\n\ntarget: rótulo do arquivo (letra a qual a imagem se refere)')
    df = pd.read_csv('./assets/map.csv')
    st.write('Tabela original')

    # Correção da coluna de rótulo
    df['target_number'] = df['target'].apply(lambda x: map_letter(x))
    st.write(df)

    # Cria a estrutura de dados 'Bunch', contendo imagens, target e target_names
    data = Bunch()
    data.images = np.array(images_r)
    data.target = df['target_number'].to_numpy()
    # st.write(data)

    # Treino do modelo
    st.header('Treinamento do Modelo')
    model_svm, model_rna = train_model(data)

    # Comparacão dos Modelos
    st.header('Comparação dos Modelos')
    st.markdown('**Score SVM:** ' + str(model_svm['score']))
    st.markdown('**Score RNA:** ' + str(model_rna['score']))

    # Teste do Modelo
    # st.header('Teste do Modelo')
    # st.warning('Atenção! Aqui é necessário que seja inserida uma imagem que o modelo não tenha visto ainda!')

    # image_p = st.file_uploader('Insira uma imagem aqui', type='png')
    # if image_p is not None:
    #     test_model(model, np.array(Image.open(image_p)))



# Método Principal para rodar o programa
if __name__ == "__main__":
    main()
