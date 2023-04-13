import pandas as pd
from io import BytesIO
import io
import base64
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.utils import img_to_array
import random
import pyrebase


firebaseConfig = {
  'apiKey': "AIzaSyDQVkry-o58GtizLatmsqTrYIOzQHUrvcc",
  'authDomain': "drawingapp1-c3dfb.firebaseapp.com",
  'projectId': "drawingapp1-c3dfb",
  'databaseURL': 'https://drawingapp1-c3dfb-default-rtdb.asia-southeast1.firebasedatabase.app',
  'storageBucket': "drawingapp1-c3dfb.appspot.com",
  'messagingSenderId': "86730802616",
  'appId': "1:86730802616:web:146f669b96023ae713fb4d",
  'measurementId': "G-8DCSSH3MTC"
}

#auth
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth

#db
db = firebase.database()
storage = firebase.storage()

def load_model():
    model = tf.keras.models.load_model('model4.h5')
    return model
model = load_model()

def preprocess(image):
    image = image.resize((350, 350))
    image_array = np.array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    image_normalized = image_batch / 255.0
    return image_normalized

def prediksi(image_data, model):
    size = (350,350)
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    imgarray = img_to_array(img)
    img = imgarray[np.newaxis,...]
    nama_class = ['delapan',
                    'dua',
                    'empat',
                    'enam',
                    'lima',
                    'nol',
                    'satu',
                    'sembilan',
                    'tiga',
                    'tujuh']
    x = imgarray
    x = np.expand_dims(x, axis=0)
    imgs = np.vstack([x])
    predik = model.predict(imgs, batch_size=100)
    for j in range(10):
        if predik[0][j]==1. :
            result = str(nama_class[j])
            return result
            break

def get_data():
    data = db.child("data").get()
    return data
        
def save_data(data, img):
    db.child("data").push(data)
    img_name = data["name"] + ".jpg"
    storage.child(img_name).put(img)

menu = ["Drawing", "Info Aplikasi", "Tutorial"]
choice = st.sidebar.selectbox("Pilih Halaman", menu)

if choice == "Drawing":
    st.title("Drawing Apps For Kids")
    st.subheader('_Belajar menulis tidak harus menggunakan buku._')

    daft_angka = ['Angka 0', 'Angka 1', 'Angka 2', 'Angka 3', 'Angka 4', 'Angka 5', 'Angka 6', 'Angka 7', 'Angka 8', 'Angka 9',]
    pilihan_soal = st.selectbox("Pilih soal:", daft_angka)


    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    canvas_result = st_canvas(
        fill_color="rgba(350, 350, 0, 0.1)",
        stroke_color=stroke_color,
        stroke_width=0.5,
        background_color=bg_color,
        update_streamlit=realtime_update,
        width=350,
        height=350,
        drawing_mode="freedraw",
        key="canvas",
    )
    if st.button("Simpan Gambar"):
        image = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("RGB")
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        img_bytes = img_buffer.getvalue()
        
        # Input data yang ingin disimpan pada database
        name = str(pilihan_soal)
        label = str(prediksi)
        data = {"name": name, "label": label}
        save_data(data, img_bytes)

    if st.button("Cek Jawaban"):
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype(np.uint8)).convert("RGB")
            processed_image = preprocess(image)
            predictions = prediksi(image, model)
            valid = str(pilihan_soal)
            if valid == "Angka 0":
                if predictions == "nol":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 3":
                if predictions == "tiga":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 5":
                if predictions == "lima":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 6":
                if predictions == "enam":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            elif valid == "Angka 7":
                if predictions == "tujuh":
                    st.write("Gambar diprediksi sebagai ",predictions, ", dan gambar merupakan ", valid,". Jawaban Kamu Benar!")
                else:
                    st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
            else:
                st.write("Gambar diprediksi sebagai ",predictions, ", namun sebenarnya gambar merupakan ", valid)
        else:
            st.write("Harap gambar dihasilkan terlebih dahulu menggunakan canvas.")

elif choice == "Info Aplikasi":
    st.title("Info Aplikasi")
    info_text = """
    ## Aplikasi ini merupakan aplikasi latihan menulis angka 0 - 9 bagi anak pra-sekolah.

    ## Aplikasi ini dibangun menggunakan metode CNN/Convolutional Neural Network.

    ## Aplikasi ini dibangun oleh Mikogizka Satria Kartika (1301194086) dan Dr. Putu Harry Gunawan, S.Si., M.Si., M.Sc. sebagai dosen pembimbing
    """
    st.write(info_text)

elif choice == "Tutorial":
    st.title("Tutorial")
    st.write("""
    ## Cara Menggunakan Drawing Apps For Kids
    1. Ada beberapa menu pada sidebar, yaitu "Drawing", "Info Aplikasi", dan "Tutorial".
    2. Untuk mulai menggambar pilih menu "Drawing", untuk melihat info aplikasi pilih menu "Info Aplikasi", dan untuk melihat panduan penggunaan aplikasi pilih menu "Tutorial"
    3. Pada menu Drawing, untuk menggambar, pilih soal terlebih dahulu pada menu yang tersedia.
    4. Gunakan Mouse untuk menggambar pada canvas yang tersedia, sesuai dengan soal yang telah anda pilih.
    5. Klik tombol Cek Jawaban untuk mengecek jawaban yang telah anda gambar/tulis pada canvas.
    6. Setelah klik tombol Cek Jawaban, maka akan dioutputkan hasil pengecekan jawabannya.
    7. Klik tombol Simpan Gambar untuk menyimpan jawaban yang telah anda gambar/tulis pada canvas ke dalam database.
    8. Setelah klik tombol Simpan Gambar, maka record jawaban anda akan tersimpan di database.
""")