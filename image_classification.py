import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


# 画像(img)が属するクラスを推論する関数（この関数は変更する必要なし）
def teachable_machine_classification(img, weights_file):

    # モデルの読み込み
    model = load_model(weights_file)

    # kerasモデルに投入するのに適した形状の配列を作成する。
    # 配列に入れることができる画像の「長さ」または枚数は
    # shapeタプルの最初の位置（この場合は1）で決まる。
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # これを画像へのパスに置き換える
    # image = Image.open(img)
    image = img

    # Teachable Machineと同じ方法で、224x224にリサイズする。
    # 少なくとも224x224になるように画像をリサイズし、中心から切り取る。
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # 画像をnumpyの配列に変換する
    image_array = np.asarray(image)

    # 画像の正規化
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 画像を配列に読み込む
    data[0] = normalized_image_array

    # 推論を実行する
    prediction = model.predict(data)

    # 推論結果をメインモジュールに戻す
    return prediction.tolist()[0]


# メインモジュール
def main():

    # タイトルの表示
    st.title("Image Classification with Google's Teachable Machine")

    # アップローダの作成
    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")

    # 画像がアップロードされた場合...
    if uploaded_file is not None:

        # 画像を画面に表示
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # teachable_machine_classification関数に画像を引き渡してクラスを推論する
        prediction = teachable_machine_classification(image, 'keras_model.h5')        
        st.caption(f'推論結果：{prediction}番') # 戻り値の確認（デバッグ用）

        classNo = np.argmax(prediction)          # 一番確率の高いクラス番号を算出
        st.caption(f'判定結果：{classNo}番')      # 戻り値の確認（デバッグ用）

        # 推論の確率を小数点以下3桁で丸め×100(%に変換)
        pred0 = round(prediction[0],3) * 100  # きのこの山の確率(%)
        pred1 = round(prediction[1],3) * 100  # たけのこの里の確率(%)

        # 推論で得られたクラス番号(初期値は0)によって出力結果を分岐
        if classNo == 0:
            st.subheader(f"これは{pred0}％の確率で「きのこの山」です！")
        else:
            st.subheader(f"これは{pred1}％の確率で「たけのこの里」です！")


# mainの起動
if __name__ == "__main__":
    main()
