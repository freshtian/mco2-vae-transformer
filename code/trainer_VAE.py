import os, warnings

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from TimeVAE import TimeVAE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import tensorflow as tf


# import utils


class VAETrainer():
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(VAETrainer.DEFAULTS, **config)

        self.data_dir = './final2/'
        self.train_file = 'train.npy'
        self.test_file = 'test.npy'

        self.win_size = 24
        self.n_feature = 5
        self.batch_size = 32

        self.train, self.val, self.test = self.get_data(self.data_dir, self.train_file, self.test_file)

        early_stop_loss = 'loss'
        self.early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta=1e-4, patience=10)
        self.reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)

    def training(self, epochs=1000):
        # 定义训练过程，包括不同潜在维度和隐藏层大小的模型训练
        latent_dims = [ 64,32,128]  # 减少潜在空间维度
        hidden_layer_sizes_list = [[512,512]]  # 减少隐藏层大 小
        # hidden_layer_sizes_list = [[512, 512]]
        losses = np.zeros((len(latent_dims), len(hidden_layer_sizes_list)))
        min_loss = 100000
        # 遍历不同的潜在空间维度
        for i, latent_dim in enumerate(latent_dims):
            # 遍历不同的隐藏层大小
            for j, hidden_layer_sizes in enumerate(hidden_layer_sizes_list):
                # 清除Keras后端会话，避免内存泄漏
                tf.keras.backend.clear_session()
                # 创建一个新的TimeVAE模型实例
                model = TimeVAE(seq_len=self.win_size, feat_dim=self.n_feature, latent_dim=latent_dim,
                                hidden_layer_sizes=hidden_layer_sizes)
                # 编译模型，使用Adam优化器
                model.compile(optimizer=Adam())
                # 训练模型并记录训练历史
                history = model.fit(
                    self.train,
                    validation_split=0.2, # 20%的数据用于验证
                    batch_size=self.batch_size,
                    epochs=epochs,
                    shuffle=True,# 每轮训练前打乱数据
                    callbacks=[self.early_stop_callback, self.reduceLR],# 使用提前停止和学习率衰减的回调函数
                    verbose=1
                )
                losses[i, j] = history.history['loss'][-1]
                if min_loss > losses[i, j]:
                    min_loss = losses[i, j]
                    min_i = i
                    min_j = j

                model_dir = f'./final2_vae_model/latent_dim_{latent_dim}_layers_{len(hidden_layer_sizes)}/'
                os.makedirs(model_dir, exist_ok=True)
                model_file_pref = f'model_'
                model.save(model_dir, model_file_pref)

                plt.clf()
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Training/Validation Loss')
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.legend(['train loss', 'validation loss'])
                plt.grid(True)
                plt.savefig(f'{model_dir}loss.png', dpi=500)
                # 使用当前模型预测训练数据
                recon = model.predict(self.train)
                self.draw_orig_and_recon_sample(self.train, recon, 3, f'{model_dir}sample.png')
                # 绘制原始数据和重构数据的样本图
                self.draw_orig_and_recon_timeseries(self.train, recon, 0, f'{model_dir}series.png')

        print('losses :\n', losses)
        print(min_loss, min_i, min_j)

    def draw_orig_and_recon_sample(self, orig, recon, n, filename):
        # 创建一个包含2行n列子图的图表
        fig, axs = plt.subplots(2, n)
        for i in range(n):
            # 随机选择一个样本的索引
            rnd_idx = np.random.choice(len(orig))
            o = orig[rnd_idx]
            r = recon[rnd_idx]
            im0 = axs[0][i].imshow(o, cmap=cm.gray, origin='lower', vmin=-2, vmax=4)
            fig.colorbar(im0, ax=axs[0][i])
            axs[0][i].set_title(f'original data #{i}', fontsize=10)
            axs[0][i].set_xlabel('feature')
            axs[0][i].set_ylabel('hour')

            im1 = axs[1][i].imshow(r, cmap=cm.gray, origin='lower', vmin=-2, vmax=4)
            fig.colorbar(im1, ax=axs[1][i])
            axs[1][i].set_title(f'reconstructed #{i}', fontsize=10)
            axs[1][i].set_xlabel('feature')
            axs[1][i].set_ylabel('hour')

        fig.suptitle("Comparison between original and reconstructed data")
        fig.tight_layout()
        plt.savefig(filename, dpi=500)

    def draw_orig_and_recon_timeseries(self, orig, recon, feature, filename):

        fig, (ax1, ax2) = plt.subplots(2, 1)
        o = orig[:, 0, feature].flatten()
        ax1.plot(o)
        ax1.set_title('original')
        r = recon[:, 0, feature].flatten()
        ax2.plot(r)
        ax2.set_title('reconstructed')
        fig.suptitle(f'Original vs Reconstructed time series of {feature}th feature')
        fig.tight_layout()
        plt.savefig(filename, dpi=500)

    def draw_orig_and_anomaly_score_timeseries(self, orig, score, feature, filename, x=None):

        fig, (ax1, ax2) = plt.subplots(2, 1)
        o = orig[:, 0, feature].flatten()
        ax1.plot(x, o) if x else ax1.plot(o)
        ax1.set_title('original')
        r = score
        ax2.plot(x, r) if x else ax2.plot(r)
        ax2.set_title('anomaly score')
        fig.suptitle(f'Original of {feature}th feature & anomaly score')
        fig.tight_layout()
        plt.savefig(filename, dpi=500)

    def get_data(self, dir, train_file, test_file):

        scaler = StandardScaler()

        train_data = np.load(dir + train_file)

        scaler.fit(train_data)
        train_data = scaler.transform(train_data)

        test_data = np.load(dir + test_file)
        test_data = scaler.transform(test_data)

        data_len = train_data.shape[0]
        val_data = train_data[(int)(data_len * 0.8):]

        train_win = self.rolling_windows(train_data, self.win_size)
        val_win = self.rolling_windows(val_data, self.win_size)
        test_win = self.rolling_windows(test_data, self.win_size)

        return train_win, val_win, test_win

    def rolling_windows(self, data, win_size):

        n_window = len(data) - win_size + 1
        window = np.zeros((n_window, win_size, self.n_feature))
        for i in range(0, n_window):
            window[i] = data[i:i + win_size]
        return window

    def plot_window(self, ax, window):

        ax.imshow(window, cmap=cm.gray)
        # ax.colorbar()
        ax.show()

    @staticmethod
    def load_model(model_dir, model_file_pref):
        params_file = os.path.join(model_dir, f'{model_file_pref}parameters.pkl')
        dict_params = joblib.load(params_file)

        model = TimeVAE(**dict_params)
        model.load_weights(model_dir, model_file_pref)
        model.compile(optimizer=Adam(learning_rate=1e-5))
        # model.summary()
        return model

    def get_anomaly_score(self, prior, recon):
        """
           计算原始数据和重构数据之间的异常得分。

           参数:
           prior (np.ndarray): 原始数据的潜在空间表示（编码后的数据）。
           recon (np.ndarray): 重构数据的潜在空间表示（解码后的数据）。

           返回:
           float: 原始数据和重构数据之间的异常得分。
           """
        # 计算原始数据和重构数据之间的均方误差（MSE）
        # np.square(prior, recon) 计算 (prior - recon) 的平方
        # np.sum(...) 计算这些平方差的总和
        s = np.sum(np.square(prior, recon))
        return s

    def test_anomaly(self, data, latent_dim=64, hidden_layer_sizes=[512, 512]):
        model_dir = f'./final2_vae_model/latent_dim_{latent_dim}_layers_{len(hidden_layer_sizes)}/'
        model_file_pref = f'model_'
        loaded_model = self.load_model(model_dir, model_file_pref)

        test_data = data
        z_mean, z_var, encoded = loaded_model.encoder(test_data)
        decoded = loaded_model.decoder(encoded)
        decoded = decoded.numpy()

        np.save('final2/new_encoded_data.npy', encoded)

        score = np.zeros(len(test_data))
        for i in range(len(test_data)):
            score[i] = self.get_anomaly_score(test_data[i], decoded[i])
            # print(i, score[i])
        np.save('anomaly_score', score)

        x_timestamp = pd.date_range('2020-08-29 0:00', periods=6625, freq='H').to_pydatetime().tolist()
        self.draw_orig_and_anomaly_score_timeseries(test_data, score, 9, 'final2_vae_model/anomaly_score_graph.png', x=x_timestamp)

    def test_load(self, latent_dim=8, hidden_layer_sizes=[512, 512], epochs=1000):
        model_dir = f'./final2_vae_model/latent_dim_{latent_dim}_layers_{len(hidden_layer_sizes)}/'
        model_file_pref = f'model_'
        loaded_model = self.load_model(model_dir, model_file_pref)
        decoded_loaded = loaded_model.predict(self.train)
        anomaly_score = np.square(self.train - decoded_loaded)
        print('Preds from orig and loaded models equal: ', np.allclose(self.train, decoded_loaded, atol=1e-4))
        self.draw_orig_and_recon_timeseries(self.train, anomaly_score, 9, 'final2_vae_model/2.png')


if __name__ == '__main__':
    trainer = VAETrainer(config=dict())
    trainer.training(epochs=1000)
    # trainer.test_load(latent_dim = 128, hidden_layer_sizes=[512,512])
    # trainer.test_anomaly(trainer.test)
