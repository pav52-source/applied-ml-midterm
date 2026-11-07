"""Report tables and calibration plot from measured artifacts; no training."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'writeup'


def table(path,columns,rows):
    lines=['\\begin{tabular}{'+columns+'}',r'\toprule']
    for i,row in enumerate(rows):
        lines.append(' & '.join(map(str,row))+r' \\')
        if i==0:lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    path.write_text('\n'.join(lines)+'\n')


def main():
    (REPORTS/'tables').mkdir(parents=True,exist_ok=True)
    (REPORTS/'figures').mkdir(exist_ok=True)
    metrics=pd.read_csv(ROOT/'part2/calibration_outputs/validation_metrics.csv')
    titles={'baseline':'Raw','sigmoid_pooled':'Pool','sigmoid_by_phase':'Phase'}
    rows=[['Calibration','Row loss','Phase mean','Brier']]
    for candidate,title in titles.items():
        overall=metrics.loc[metrics.candidate.eq(candidate)&metrics.phase.eq('overall')].iloc[0]
        mean=metrics.loc[metrics.candidate.eq(candidate)&metrics.phase.ne('overall')].log_loss.mean()
        rows.append([title,f'{overall.log_loss:.5f}',f'{mean:.5f}',f'{overall.brier_score:.5f}'])
    table(REPORTS/'tables/overall.tex','lrrr',rows)
    rows=[['Phase','Cal.','Loss','Brier','F1','ROC','AP']]
    for phase in ('I','II','III'):
        for candidate,title in titles.items():
            row=metrics.loc[metrics.candidate.eq(candidate)&metrics.phase.eq(phase)].iloc[0]
            rows.append([phase,title,*[f'{row[m]:.4f}' for m in ('log_loss','brier_score','f1','roc_auc','average_precision')]])
    table(REPORTS/'tables/phase_metrics.tex','llrrrrr',rows)
    groups=pd.read_csv(ROOT/'integration/outputs/reason_groups.csv')
    rows=[['Phase','Reason','n','T','P','Raw','Pool']]
    for phase in ('I','II','III'):
        raw=groups.loc[groups.candidate.eq('baseline')&groups.phase.eq(phase)].set_index('reason_group')
        cal=groups.loc[groups.candidate.eq('sigmoid_pooled')&groups.phase.eq(phase)].set_index('reason_group')
        for reason,r in raw.iterrows():
            name='No record' if reason=='no recorded stopping reason' else reason
            raw_rate='--' if not r.n_trials else f'{r.false_success_rate:.3f}'
            cal_rate='--' if not r.n_trials else f'{cal.loc[reason,"false_success_rate"]:.3f}'
            rows.append([phase,name,int(r.n_trials),int(r.n_ground_truth),int(r.n_predictions),raw_rate,cal_rate])
    table(REPORTS/'tables/reason_errors.tex','llrrrrr',rows)
    bins=pd.read_csv(ROOT/'part2/calibration_outputs/validation_calibration_bins.csv')
    fig,ax=plt.subplots(figsize=(4.4,3.1))
    ax.plot([0,1],[0,1],'--',color='gray',label='Ideal')
    for name,title in [('baseline','Raw'),('sigmoid_pooled','Pooled sigmoid')]:
        b=bins.loc[bins.phase.eq('III')&bins.candidate.eq(name)&bins.n.gt(0)]
        ax.plot(b.mean_probability,b.observed_success_rate,'o-',label=title)
    ax.set(xlabel='Mean predicted success',ylabel='Observed success fraction',xlim=(0,1),ylim=(0,1),title='Phase III validation reliability')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(REPORTS/'figures/phase3_reliability.pdf')
    plt.close(fig)
    print('Generated three measured tables and the reliability figure; no models fitted.')


if __name__=='__main__':main()
