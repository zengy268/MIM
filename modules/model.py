import torch.autograd
from torch import nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPreTrainedModel
from modules.transformer import TransformerEncoder
from modules.bert import BertModel
from global_configs import config as config1
import numpy as np
from torch.nn.parameter import Parameter
from torch import sigmoid

class MIM(nn.Module):
    def __init__(self):
        super(MIM, self).__init__()
        self.l_encoder = L_Encoder.from_pretrained(config1.model_choice, num_labels=1)
        self.a_encoder = A_Encoder()
        self.v_encoder = V_Encoder()
        
        self.l_mask = MaskLearner(sequence_length=49)
        self.a_mask = MaskLearner()
        self.v_mask = MaskLearner()
        
        self.l_classifier = Classifier()
        self.a_classifier = Classifier()
        self.v_classifier = Classifier()
        
        self.fusion = Fusion()
    
    def forward(self, l, a, v):
        return l+a+v
        
class L_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.activate = config1.activate
        self.d_l = config1.d_l
        self.input_size = config1.d_l
        self.proj_l = nn.Conv1d(768, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )  

        outputs = outputs.transpose(1, 2)
        outputs = self.proj_l(outputs)  
        pooled_output = self.pooler(self.activate(outputs)).squeeze(dim=2)

        return pooled_output, outputs.permute(2, 0, 1) 


class A_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_l = config1.d_l
        self.proj_a = nn.Conv1d(config1.ACOUSTIC_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_size = config1.d_l
        self.pooler = nn.AdaptiveAvgPool1d(1)
        # self.activate = nn.ReLU(inplace=False)
        # self.activate = nn.Tanh()
        self.activate = config1.activate
        self.transa = TransformerEncoder(embed_dim=config1.d_l,
                              num_heads=5, #self.num_heads,
                              layers=3,#max(self.layers, layers),
                              attn_dropout=0.5,
                              relu_dropout=0.3,#self.relu_dropout,
                              res_dropout= 0.3,#self.res_dropout,
                              embed_dropout=0.2,#self.embed_dropout,
                              attn_mask= False)#self.attn_mask)

    def forward(self, acoustic):
        # input: (bs, ts, dim)
        acoustic = self.proj_a(acoustic.transpose(1, 2))
        acoustic = acoustic.permute(2, 0, 1)
        # acoustic = self.activate(acoustic)
        outputa = self.transa(acoustic)
        pooled_output = self.pooler(self.activate(outputa.permute(1, 2, 0))).squeeze(dim=2)  # (bs, dim)

        return pooled_output, outputa


class V_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_l = config1.d_l
        self.proj_v = nn.Conv1d(config1.VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_size = config1.d_l
        self.pooler = nn.AdaptiveAvgPool1d(1)
        # self.activate = nn.ReLU(inplace=False)
        # self.activate = nn.Tanh()
        self.activate = config1.activate
        self.transv = TransformerEncoder(embed_dim=config1.d_l,
                              num_heads=5, #self.num_heads,
                              layers=3,#max(self.layers, layers),
                              attn_dropout=0.5,
                              relu_dropout=0.3,#self.relu_dropout,
                              res_dropout= 0.3,#self.res_dropout,
                              embed_dropout=0.2,#self.embed_dropout,
                              attn_mask= False)#self.attn_mask)

    def forward(self, visual):
        # input: (bs, ts, dim)
        visual = self.proj_v(visual.transpose(1, 2))
        # visual = self.activate(visual)
        visual = visual.permute(2, 0, 1)
        outputv = self.transv(visual)  # transformer input (ts, bs, dim)
        pooled_output = self.pooler(self.activate(outputv.permute(1, 2, 0))).squeeze(dim=2)  # (bs, dim)

        return pooled_output, outputv  # unimodal encoding # (bs, dim) & (ts, bs, dim


class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1/3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3): #  beta=1/3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True):
        input_element = input_element + self.loc_bias

        u = torch.empty_like(input_element).uniform_(1e-6, 1.0-1e-6)
        s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)
        penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
        s = s * (self.zeta - self.gamma) + self.gamma

        if summarize_penalty:
            penalty = penalty.mean()

        clipped_s = self.clip(s)

        if True:
            hard_concrete = (clipped_s > 0.6).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)


class MaskLearner(nn.Module):
    def __init__(self, sequence_length=50):
        super(MaskLearner, self).__init__()
        self.d_l = config1.d_l
        self.lstm = nn.LSTM(input_size=self.d_l, hidden_size=self.d_l // 2, num_layers=1, bidirectional=True)
        self.mask = HardConcrete()
        self.ratio = 5
        # self.activate = nn.Tanh()
        # self.activate = nn.ReLU()
        self.activate = config1.activate
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.pooler2 = nn.Linear(self.d_l, 1)
        self.pooler3 = nn.Sequential(
            nn.Linear(self.d_l, self.d_l // 2),
            nn.ReLU(), #todo
            nn.Linear(self.d_l // 2, 1),
            nn.ReLU(),
        )
        self.tanh = nn.Tanh()
        self.imp_ratio = nn.Sequential(
            nn.Linear(sequence_length, sequence_length // self.ratio),
            nn.ReLU(),
            nn.Linear(sequence_length // self.ratio, sequence_length),
        )

    def forward(self, x):
        # input: (ts, bs, dim)
        x_time, _ = self.lstm(x)
        x_time = self.pooler3(x_time.transpose(0, 1)).squeeze(2)   # squeeze dim for each ts: (bs, ts)
        x_time = self.imp_ratio(x_time)
        x_mask, penalty = self.mask(x_time)
        x_mask = x_mask.transpose(0, 1).unsqueeze(2) 
        x_mask = torch.mul(x, x_mask)  # refined output
        x_mask = self.pooler(self.activate(x_mask.permute(1, 2, 0))).squeeze(2)  # pool ts for refined output
        # x_mask = self.pooler(x_mask.permute(1, 2, 0)).squeeze(2)  # pool ts for refined output

        return x_mask, penalty


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_l = config1.d_l
        self.classifier = nn.Sequential(
            nn.Linear(self.d_l, 1)
        )

    def forward(self, x):
        output = self.classifier(x)

        return output


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.d_l = config1.d_l
        self.classifier = Classifier()
        self.linear = nn.Linear(self.d_l * 3, self.d_l)
        self.linear_mul = nn.Linear(self.d_l, self.d_l)
        self.fusion = 'multiplication'
        self.activate = nn.Tanh()

    def forward(self, l, a, v):
        if self.fusion == 'add':
            fusion_output = l + a + v
        elif self.fusion == 'multiplication':
            fusion_output = torch.tanh(self.linear_mul(l * a * v))
        elif self.fusion == 'concat':
            fusion_output = torch.cat([l, a, v], dim=-1)
            fusion_output = self.activate(self.linear(fusion_output))

        fusion_output = self.classifier(fusion_output)

        return fusion_output
