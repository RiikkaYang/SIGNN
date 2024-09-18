import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import grad

from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from modules import MLPNet
import torch.nn.functional as F

def gradients_mse(ode_in, ode_out, rhs):
    gradients = diff_gradient(ode_out, ode_in)  
    ODE_loss = (gradients - rhs).pow(2).sum(-1)
    return ODE_loss

def diff_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

loss_func = nn.CrossEntropyLoss()

def loss_function(model_output, gt, loss_definition="CE"):
    pred_opinion_label = model_output['opinion_label'] ## Predicted opinion label
    gt_latent_opinion = gt['opinion'] ## Ground truth opinion label

    ### Compute data loss $\mathcal{L}_{data}$
    data_loss = loss_func(pred_opinion_label, gt_latent_opinion[:,0].long())

    regularizer = model_output['regularizer']
    ode_constraint = model_output['ode_constraint']

    # Exp      # Lapl
    # -----------------
    return {'data_loss': data_loss, # * 1e2,
            'ode_constraint': ode_constraint.mean(),
            'regularizer_constraint': regularizer.mean()
           }

def gumbel_softmax(logits, temperature=0.2):
    eps = 1e-20
    u = torch.rand(logits.shape)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

class model(MetaModule):

    def __init__(self, num_users=1, type='relu',  
                 hidden_features=256, num_hidden_layers=3, nclasses=None, **kwargs):
        super().__init__()

        self.U = num_users  ## Number of users
        self.type_odm = kwargs["type_odm"]  ## Choice of opinion dynamics model 

        ### Set hyperparameters
        self.alpha = kwargs["alpha"]  ## Trade-off hyperparameter $\alpha$
        self.beta = kwargs["beta"]  ## Regularization parameter $\beta$
        self.K = kwargs["K"]  ## Dimension of the latent space 
        self.J = 1  ## Number of collocation points $J$

        ### Prepare user profile information
        profiles = kwargs["df_profile"]  ## Hidden user representation $\{{\bf h}_1,...,{\bf h}_U\}$
        if profiles is None:
            use_graph = False  #non use of oser profile,we don't have too many profiles
        else:
            use_graph = True
            self.profiles = profiles
            #profiles = profiles.reshape(-1,25,768)
            #self.profiles = torch.from_numpy(np.array(profiles, dtype=np.float32)).clone()
        self.use_graph = use_graph

        ### Prepare neural network $f(t,\{bf e\}_u;\theta_f)$
        self.net = MLPNet(num_users=self.U, num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, outermost_linear=type, nonlinearity=type, use_graph=use_graph)
        self.val2label = nn.Linear(1, nclasses)

        ### Prepare ODE parameters $\Lambda$ 
        if self.type_odm=="SBCM":
            self.rho = nn.Parameter(torch.ones(1))  ## Exponent parameter $\rho$ 

    def sampling(self,vec):
        vec = F.softmax(vec, dim=1)
        logits = gumbel_softmax(vec, 0.1)
        return logits

    def forward(self, model_input, params=None):
        times = model_input['ti']
        uids = model_input['ui']

        if self.use_graph:
            profs = self.profiles
            output, attention = self.net(times, uids, profs)
        else:
            output = self.net(times, uids)
            attention = None

        ### Predict opinion labels
        opinion_label = self.val2label(output)

        ### Setup ODE constraints
        tilde_z_ut = None 

        if self.training:
            tau_j = torch.rand(self.J).unsqueeze(1).requires_grad_(True)

            users = torch.arange(self.U).unsqueeze(1)
            taus = tau_j.repeat(users.shape[0],1)

            if self.use_graph:
                #_profs = torch.index_select(self.profiles,0,users[:,0])
                _profs = self.profiles
                _vector_x, _ = self.net(taus, users, _profs)
            else:
                _vector_x = self.net(taus, users)

            ## Predicted opinions of $U$ users $\{\tilde{x}_1(\tau_j),...,\tilde{x}_U(\tau_j)\}$
            vector_x = torch.transpose(torch.reshape(_vector_x, (self.U, self.J)), 1, 0)

            user_id = torch.randint(self.U-1, (1,1))  ### Sample user $u$
            if self.use_graph:
                #_profs = torch.index_select(self.profiles,0,user_id[:,0])
                _profs = self.profiles
                x_u, _ = self.net(tau_j, user_id, _profs)
            else:
                ### Predict opinion $\tilde{x}_u(\tau_j)$ of user $u$ at time $\tau_j$
                x_u = self.net(tau_j, user_id)  

            if self.type_odm=="SBCM":
                distance = torch.abs(x_u - vector_x)

                ## Probability of user $u$ selecting user $v$ as an interaction partner at time $\tau_j$
                p_uv = (distance + 1e-12).pow(self.rho)

                ## Differentiable one-hot approximation $\tilde{z}_u^t$ in Equation (9)
                tilde_z_ut = self.sampling(p_uv)

                ## Right hand side (rhs) of Equation (10)
                rhs_ode = tilde_z_ut * (x_u - vector_x)
                rhs_ode = rhs_ode.sum(-1)

                ## Regularization term $\mathcal{R}(\Lambda)$
                regularizer = self.beta * torch.zeros(1)

            rhs_ode = torch.reshape(rhs_ode, (-1,self.J))

            ### Compute ODE loss $\mathcal{L}_{ode}$
            ode_constraints = gradients_mse(tau_j, x_u, rhs_ode)
            ode_constraint = self.alpha * ode_constraints
        else:
            ode_constraint = torch.zeros(1)
            regularizer = torch.zeros(1)


        return {'opinion': output, 'opinion_label': opinion_label, 'ode_constraint': ode_constraint, 
                'regularizer': regularizer, 'attention': attention, 'zu': tilde_z_ut}