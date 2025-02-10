import torch, math

class Channels():
    def __init__(self, device):
        self.device = device

    def calculate_signal_power(self, Tx_sig):
        return torch.mean(Tx_sig ** 2)

    def AWGN(self, Tx_sig, snr, H=None):
        if H is not None:
            shape = Tx_sig.shape
            Tx_sig_without_channel = torch.matmul(Tx_sig, torch.inverse(H)).view(shape)
            signal_power = self.calculate_signal_power(Tx_sig_without_channel)
        else:
            signal_power = self.calculate_signal_power(Tx_sig)
        n_var = signal_power / 10**(snr / 10)
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(n_var), size=Tx_sig.shape).to(self.device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, snr):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(self.device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(self.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, snr, H)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, snr, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(self.device)
        H_imag = torch.normal(mean, std, size=[1]).to(self.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, snr, H)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape) 
        return Rx_sig
