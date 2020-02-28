from argparse import ArgumentParser

parser=ArgumentParser()

parser.add_argument('--inputpath', type=str, dest='inputpath', default='Input/g20/Set5/')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='GT/Set5/')
parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='Input/g20/kernel.mat')
parser.add_argument('--savepath', type=str, dest='savepath', default='results/Set5')
parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=0)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)

parser.add_argument('--gpu', type=str, dest='gpu', default='0')

args= parser.parse_args()