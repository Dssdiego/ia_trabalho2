import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import io
import os
import zipfile

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
    image_b = binarize_pixel(image_matrix=image_src, thresh=128)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(2, 4))

    ax1.axis("off")
    ax1.title.set_text('Original')

    ax2.axis("off")
    ax2.title.set_text("Convertido")

    ax1.imshow(image_src, cmap='gray')
    ax2.imshow(image_b, cmap='gray')

    return image_b


# Carrega as imagens da pasta assets/images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        image_src = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img is not None:
            images.append(image_src)
    return images


# Treina o modelo
def train_images(images):
    # Convertendo imagens para preto e branco
    images_b = []
    for image in images:
        images_b.append(binarize(image))
    st.write('Originais (64x64)')
    st.image(images)
    st.write('Convertidos (64x64)')
    st.image(images_b)

    # Reduz os blocos das imagens (64x64 >> 8x8)
    images_r = []
    for image_b in images_b:
        images_r.append(reduce_blocks(image_b))

    st.write('Reduzidas (8x8)')
    st.image(images_r)
    # st.write(images_r)


    return


# Faz o mapeamento de uma letra para um índice
# A = 0
# B = 1
# C = 2
# D = 3
# ... etc
def map_letter(letter):
    return ord(letter.lower()) - 97


# Prevê a letra conforme a imagem passada
# usando o modelo treinado anteriormente
def predict_letter(image_src):
    # TODO: Implementar
    return 'A'


# Método principal
def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.title('Trabalho 2 - Inteligência Artificial')
    st.header('Como usar o App')
    
    st.info('As imagens devem estar disponíveis na pasta "assets/images" no formato ".png" e tamanho 64x64, numeradas de 0 a n.\n\nEx.: 0.png, 1.png, ..., n.png')

    # Cria o menu lateral
    menu = st.sidebar

    debug = menu.checkbox('Modo de Debug')

    # Lê todos os arquivos de imagem dentro da pasta asseto
    images = load_images_from_folder('./assets/images')
    # st.image(images)

    train_images(images)
    
    # Leitura do arquivo de mapeamento (imagens > letras)
    st.header('Tabela de mapeamento')
    st.info('Esta tabela mapeia os arquivos de imagem para seus devidos rótulos\n\nimage_file: nome do arquivo (sem o formato)\n\ntarget: rótulo do arquivo (letra a qual a imagem se refere)')
    df = pd.read_csv('./assets/map.csv')
    st.write('Tabela original (Resumo)')
    st.write(df.head())

    # Correção da coluna de rótulo
    st.write('Conversão da tabela - Mapeamento letras -> números (Resumo)')
    df['target'] = df['target'].apply(lambda x: map_letter(x))
    st.write(df.head())

    # Cria a estrutura de dados 'Bunch'
    b = Bunch()


    # Adição de coluna com informações das imagens
    

    # Treino do modelo
    st.header('Treinamento do Modelo')

    # Acurácia
    st.header('Acurácia do Modelo')

    # Teste do Modelo
    st.header('Teste do Modelo')
    st.warning('Atenção! Aqui é necessário que seja inserida uma imagem que o modelo não tenha visto ainda!')

    img_pred = st.file_uploader('Insira uma imagem aqui')
    if img_pred is not None:
        st.success('Esta imagem é a letra ' + predict_letter(img_pred))


# Método Principal para rodar o programa
if __name__ == "__main__":
    main()
