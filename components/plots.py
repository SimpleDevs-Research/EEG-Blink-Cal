import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.transforms as mtransforms
from . import muse as MS
from . import vr as VR
from . import trials as TR
from . import session as SS
from . import participants as PS
from . import config as CF

def plot_muse(muse:MS.Muse, config:CF.Config, parent_spec=None, show:bool=False, outname:str=None):
    # 2 rows, 3 columns.
    # first 2x2 is EEG. 1-3 is the IMU
    if parent_spec is None:
        fig = plt.figure(figsize=(15,5))
        spec = fig.add_gridspec(1, 1)[0]
    else:
        fig = parent_spec.get_gridspec().figure
        spec = parent_spec
    # Generate the grid spec.
    # We generate a 2-row, 1-col grid. Top = eeg, Bottom = IMU
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=0.1)
    # Define axes
    ax_eeg = fig.add_subplot(gs[0])
    ax_imu = fig.add_subplot(gs[1], sharex=ax_eeg)
    # Plot
    ax_eeg.plot(
        muse.raw['unix_ms'], 
        muse.raw[config.files['muse']['eeg_colname']],
        color=config.files['muse']['eeg_color'],
        label=f"Muse: {config.files['muse']['eeg_colname']}"
    )
    ax_imu.plot(
        muse.imu['unix_ms'],
        muse.imu[config.files['muse']['imu_colname']],
        color=config.files['muse']['imu_color'],
        label=f"Muse: {config.files['muse']['imu_colname']}"
    )
    # Modify axes and legend
    ax_eeg.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
    ax_eeg.legend()
    ax_imu.legend()
    # if this is an independent figure, we must save it, if needed
    if parent_spec is None:
        if outname is not None:
            outpath = os.path.join(os.path.dirname(muse.src), 'muse.png')
            plt.savefig(outpath, bbox_inches="tight", dpi=300)
        if show:    plt.show()
        else:       plt.close()
    # return axes
    return ax_eeg, ax_imu

def plot_vr(vr:VR.VR, config:CF.Config, parent_spec=None, show:bool=False, outname:str=None):
    # 2 rows, 3 columns.
    # first 2x2 is EEG. 1-3 is the IMU
    if parent_spec is None:
        fig = plt.figure(figsize=(15,5))
        spec = fig.add_gridspec(1, 1)[0]
    else:
        fig = parent_spec.get_gridspec().figure
        spec = parent_spec
    # Generate the grid spec.
    # We generate a 2-row, 1-col grid. Top = eeg, Bottom = IMU
    gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, hspace=0.1)
    # Define axes
    ax_eye = fig.add_subplot(gs[0])
    ax_imu = fig.add_subplot(gs[1], sharex=ax_eye)
    # Plot
    ax_eye.plot(
        vr.eye['unix_ms'], 
        vr.eye[config.files['vr_eye']['colname']],
        color=config.files['vr_eye']['color'],
        label=f"VR: {config.files['vr_eye']['colname']}"
    )
    ax_imu.plot(
        vr.imu['unix_ms'],
        vr.imu[config.files['vr_imu']['colname']],
        color=config.files['vr_imu']['color'],
        label=f"VR: {config.files['vr_imu']['colname']}"
    )
    # Modify axes and legend
    ax_eye.tick_params(axis="x", which="both", bottom=False, labelbottom=False )
    ax_eye.legend()
    ax_imu.legend()
    # if this is an independent figure, we must save it, if needed
    if parent_spec is None:
        if outname is not None:
            outpath = os.path.join(os.path.dirname(vr.imu_src), 'muse.png')
            plt.savefig(outpath, bbox_inches="tight", dpi=300)
        if show:    plt.show()
        else:       plt.close()
    # return axes
    return ax_eye, ax_imu

def plot_trial(trial:TR.Trial, config:CF.Config, parent_spec=None, show:bool=False, outname:str=None):
    if parent_spec is None:
        fig = plt.figure(figsize=(15,15))
        spec = fig.add_gridspec(1, 1)[0]
    else:
        fig = parent_spec.get_gridspec().figure
        spec = parent_spec
    # Generate the grid spec.
    # We generate a 1-row, 2-col grid. left = Muse, right = VR
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.1)
    # Plot Muse and VR
    ax_muse_eeg, ax_muse_imu = plot_muse(trial.muse, config, parent_spec=gs[0])
    ax_vr_eye, ax_vr_imu = plot_vr(trial.vr, config, parent_spec=gs[1])
    # As this is a trial, let's plot our calibrations as vertical lines
    for ax_i, ax in enumerate([ax_muse_eeg, ax_muse_imu, ax_vr_eye, ax_vr_imu]):
        ax_trans = mtransforms.blended_transform_factory(
            ax.transData,   # x in data coordinates
            ax.transAxes    # y in axes coordinates
        )
        for row_index, row in trial.calibration.overlaps.iterrows():
            ax.axvline( x=row['unix_ms'], color='black', alpha=0.75)
            ax.text( row['unix_ms'], 0.02, f"Start {row['overlap_counter']}",
                transform=ax_trans, rotation=90, ha='left', 
                va='bottom', fontsize=6, rotation_mode='anchor')
    # if this is an independent figure, we must save it, if needed
    if parent_spec is None:
        if outname is not None:
            outpath = os.path.join(os.path.dirname(trial.src), 'trial.png')
            plt.savefig(outpath, bbox_inches="tight", dpi=300)
        if show:    plt.show()
        else:       plt.close()
    # return axes
    return ax_muse_eeg, ax_muse_imu, ax_vr_eye, ax_vr_imu

def plot_session(session:SS.Session, config:CF.Config, parent_spec=None, show:bool=False, outname:str=None):
    if parent_spec is None:
        fig = plt.figure(figsize=(15, len(session.trials)*2))
        spec = fig.add_gridspec(1, 1)[0]
    else:
        fig = parent_spec.get_gridspec().figure
        spec = parent_spec
    # Generate the grid spec.
    # We generate a n-row, 1-col grid. Each row is a trial
    gs = GridSpecFromSubplotSpec(len(session.trials), 1, subplot_spec=spec, hspace=0.2)
    # Plot
    for i, t in enumerate(session.trials):
        plot_trial(t, config, parent_spec=gs[i]) 
    # Title
    title = f"Session: {os.path.basename(session.src)}"
    bbox = spec.get_position(fig)
    title_x = bbox.x0
    if parent_spec is None: 
        title_x += bbox.width/2
    fig.text(
        title_x,
        bbox.y1 + 0.0025,
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )
    # if this is an independent figure, we must save it, if needed
    if parent_spec is None:
        if outname is not None:
            outpath = os.path.join(session.src, 'session.png')
            plt.savefig(outpath, bbox_inches="tight", dpi=300)
        if show:    plt.show()
        else:       plt.close()


def plot_participant(p:PS.Participant, config:CF.Config, parent_spec=None, show:bool=False, outname:str=None):
    if parent_spec is None:
        fig = plt.figure(figsize=(len(p.sessions)*10, 10))
        spec = fig.add_gridspec(1, 1)[0]
    else:
        fig = parent_spec.get_gridspec().figure
        spec = parent_spec
    # Generate the grid spec.
    # We generate a 1-row, n-col grid. Each column is a session
    gs = GridSpecFromSubplotSpec(1, len(p.sessions), subplot_spec=spec, wspace=0.15)
    # Plot
    for i, s in enumerate(p.sessions):
        plot_session(s, config, parent_spec=gs[i]) 
    # Title
    title = os.path.basename(p.src)
    bbox = spec.get_position(fig)
    title_x = bbox.x0
    if parent_spec is None: 
        title_x += bbox.width/2
    fig.text(
        title_x,
        bbox.y1 + 0.0025,
        title,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold"
    )
    # if this is an independent figure, we must save it, if needed
    if parent_spec is None:
        if outname is not None:
            outpath = os.path.join(p.src, 'participant.png')
            plt.savefig(outpath, bbox_inches="tight", dpi=300)
        if show:    plt.show()
        else:       plt.close()

def plot_raws(participants, config:CF.Config, show:bool=True, outpath:str=None):
    # P-rows, 1-col. Rows represent participants
    fig = plt.figure(figsize=(5*len(config.expected_sessions),5*len(participants)), layout=None)
    # Gridspec
    gs = fig.add_gridspec(len(participants), 1, hspace=0.3, top=0.97 )
    # Plotting
    for i, p in enumerate(participants):
        plot_participant(p, config, parent_spec=gs[i])
    # Results
    if outpath is not None:
        plt.savefig(outpath, bbox_inches="tight", dpi=300)
    if show:    plt.show()
    else:       plt.close()
