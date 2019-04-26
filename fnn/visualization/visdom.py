# module visualizations.py
from datetime import datetime
import visdom

class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)

        self.loss_win = None
        self.acc_win = None
    
    def plot_example(self):
        trace = dict(
            x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
            marker={'color': 'red', 'symbol': 104, 'size': "10"},
            text=["one", "two", "three"], name='1st Trace')
        trace1 = dict(
            x=[10, 20, 30], y=[40, 50, 60], mode="markers+lines", type='custom',
            marker={'color': 'blue', 'symbol': 100, 'size': "8"},
            text=["one", "two", "three"], name='1st Trace')
        layout = dict(
            title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

        self.vis._send(
            {'data': [trace, trace1], 'layout': layout, 'win': 'mywin'})

    
    def plot_loss_train(self, loss, step):
        self.loss_win = self.vis.line(
            [loss], [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            name='Train',
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Train Loss',
                legend=['Train', 'Valid']
            )
        )


    def plot_loss_valid(self, loss, spoch):
        self.loss_win = self.vis.line(
            [loss],
            [spoch],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            name='Valid',
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Valid Loss ',
                legend=['Train', 'Valid']
            )
        )
    
    def plot_acc(self, acc, spoch):
        self.acc_win = self.vis.line(
            [acc], [spoch],
            win=self.acc_win,
            update='append' if self.acc_win else None,
            name='Valid',
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy %',
                title='Valid Accuracy',
                legend=['Valid']
            )
        )
