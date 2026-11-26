import os
import tempfile

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import plotly.graph_objects as go
from matplotlib import animation  # type: ignore

from src.schemas.training_state import EpochState


def create_animated_figure(frames_data: list[EpochState], duration: int) -> go.Figure:
    """Создает анимированный график Plotly с корректной синхронизацией треков."""
    if not frames_data:
        return go.Figure()

    first_frame = frames_data[0]
    is_gdbt = first_frame.model_type == "gdbt"

    y_min_val, y_max_val = first_frame.y_target.min(), first_frame.y_target.max()
    margin = (y_max_val - y_min_val) * 0.5 + 0.5
    y_range = [y_min_val - margin, y_max_val + margin]

    base_traces = []

    # 1. линии базовых элементов
    for i in range(first_frame.element_outputs.shape[0]):
        base_traces.append(
            go.Scatter(
                x=first_frame.x_grid,
                y=first_frame.element_outputs[i],
                mode="lines",
                line={"width": 1},
                opacity=0.35,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 2. целевая функция
    base_traces.append(
        go.Scatter(
            x=first_frame.x_grid,
            y=first_frame.y_target,
            mode="lines",
            name="Целевая f(x)",
            line={"color": "#FFFFFF", "width": 2.5, "dash": "dash"},
        )
    )

    # 3. предсказание (Train)
    y_tr_0 = np.where(first_frame.train_mask, first_frame.y_pred, np.nan)
    base_traces.append(
        go.Scatter(
            x=first_frame.x_grid,
            y=y_tr_0,
            mode="lines",
            name="Предсказание (Train)",
            line={"color": "#00D2FF", "width": 3.5},
        )
    )

    # 4. экстраполяция (Val)
    has_val = np.any(~first_frame.train_mask)
    if has_val:
        y_val_0 = np.where(~first_frame.train_mask, first_frame.y_pred, np.nan)
        base_traces.append(
            go.Scatter(
                x=first_frame.x_grid,
                y=y_val_0,
                mode="lines",
                name="Экстраполяция (Val)",
                line={"color": "#FF4B4B", "width": 2.5, "dash": "dot"},
            )
        )

    plotly_frames = []
    slider_steps = []

    for frame in frames_data:
        frame_traces = []

        # 1. деревья или нейроны
        for i in range(frame.element_outputs.shape[0]):
            frame_traces.append(go.Scatter(x=frame.x_grid, y=frame.element_outputs[i]))

        # 2. целевая функция
        frame_traces.append(go.Scatter(x=frame.x_grid, y=frame.y_target))

        # 3. предсказание (Train)
        y_tr = np.where(frame.train_mask, frame.y_pred, np.nan)
        frame_traces.append(go.Scatter(x=frame.x_grid, y=y_tr))

        # 4. экстраполяция (Val)
        if has_val:
            y_val = np.where(~frame.train_mask, frame.y_pred, np.nan)
            frame_traces.append(go.Scatter(x=frame.x_grid, y=y_val))

        frame_name = f"epoch_{frame.epoch}"
        plotly_frames.append(go.Frame(data=frame_traces, name=frame_name))

        slider_steps.append(
            {
                "args": [
                    [frame_name],
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": str(frame.epoch),
                "method": "animate",
            }
        )

    fig = go.Figure(data=base_traces, frames=plotly_frames)

    fig.add_vrect(
        x0=first_frame.x_grid[first_frame.train_mask].min(),
        x1=first_frame.x_grid[first_frame.train_mask].max(),
        fillcolor="rgba(255, 255, 255, 0.08)",
        line={"color": "rgba(255, 255, 255, 0.3)", "width": 1, "dash": "dot"},
        annotation_text=" Зона обучения",
        annotation_position="top left",
        annotation_font={"color": "white", "size": 12},
    )

    prefix_label = "Дерево: " if is_gdbt else "Эпоха: "

    fig.update_layout(
        title={
            "text": "Анимация процесса обучения",
            "font": {"size": 20, "color": "white"},
        },
        xaxis_title="x",
        yaxis_title="y",
        yaxis={"range": y_range},
        height=650,
        margin={"l": 20, "r": 20, "t": 50, "b": 80},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.1,
                "y": -0.15,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶ Воспроизвести",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": duration, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Пауза",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16, "color": "white"},
                    "prefix": prefix_label,
                    "visible": True,
                    "xanchor": "right",
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 30},
                "len": 0.85,
                "x": 0.15,
                "y": -0.15,
                "steps": slider_steps,
            }
        ],
    )

    return fig


def export_frames_to_gif(frames_data: list[EpochState], fps: int = 20) -> bytes:
    """генерит гифку из массива кадров через матплотлиб"""
    if not frames_data:
        return b""

    first_frame = frames_data[0]
    is_gdbt = first_frame.model_type == "gdbt"

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    y_min_val, y_max_val = first_frame.y_target.min(), first_frame.y_target.max()
    margin = (y_max_val - y_min_val) * 0.5 + 0.5
    ax.set_ylim(y_min_val - margin, y_max_val + margin)
    ax.set_xlim(first_frame.x_grid.min(), first_frame.x_grid.max())

    x_train_pts = first_frame.x_grid[first_frame.train_mask]
    ax.axvspan(
        x_train_pts.min(),
        x_train_pts.max(),
        color="white",
        alpha=0.08,
        label="Зона обучения",
    )

    (_line_target,) = ax.plot(
        first_frame.x_grid,
        first_frame.y_target,
        "w--",
        linewidth=2,
        label="Целевая f(x)",
    )

    elem_lines = []
    for _ in range(first_frame.element_outputs.shape[0]):
        (line_elem,) = ax.plot([], [], linewidth=1, alpha=0.35)
        elem_lines.append(line_elem)

    (line_train,) = ax.plot([], [], color="#00D2FF", linewidth=3, label="Train")
    (line_val,) = ax.plot(
        [], [], color="#FF4B4B", linewidth=2, linestyle=":", label="Val"
    )

    prefix_label = "Дерево: " if is_gdbt else "Эпоха: "
    title_text = ax.set_title(f"{prefix_label}0", color="white", fontsize=12)

    ax.legend(loc="upper right", facecolor="#1e1e1e", edgecolor="none")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()

    def update(frame: EpochState):
        title_text.set_text(f"{prefix_label}{frame.epoch}")

        for i, line_elem in enumerate(elem_lines):
            if i < frame.element_outputs.shape[0]:
                line_elem.set_data(frame.x_grid, frame.element_outputs[i])

        y_tr = np.where(frame.train_mask, frame.y_pred, np.nan)
        y_val = np.where(~frame.train_mask, frame.y_pred, np.nan)

        line_train.set_data(frame.x_grid, y_tr)
        line_val.set_data(frame.x_grid, y_val)

        return [title_text, line_train, line_val, *elem_lines]

    anim = animation.FuncAnimation(
        fig, update, frames=frames_data, blit=False, interval=1000 // fps
    )

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    try:
        anim.save(tmp_path, writer="pillow", fps=fps)  # type: ignore

        with open(tmp_path, "rb") as f:
            gif_bytes = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        plt.close(fig)

    return gif_bytes


# ================ LEGACY ================
# отрисовка графика
# def create_figure(frame: EpochState) -> go.Figure:
#     """Делает figure из стейта эпохи сетки"""
#     fig = go.Figure()

#     y_min_val, y_max_val = frame.y_target.min(), frame.y_target.max()
#     margin = (y_max_val - y_min_val) * 0.5 + 0.5
#     y_range = [y_min_val - margin, y_max_val + margin]

#     # fig.add_vrect(
#     #     x0=frame.x_grid[frame.train_mask].min(),
#     #     x1=frame.x_grid[frame.train_mask].max(),
#     #     fillcolor="rgba(255, 255, 255, 0.08)",
#     #     line={"color": "rgba(255, 255, 255, 0.3)", "width": 1, "dash": "dot"},
#     #     annotation_text=" Зона обучения",
#     #     annotation_position="top left",
#     #     annotation_font={"color": "white", "size": 12},
#     # )

#     for i in range(frame.element_outputs.shape[0]):
#         fig.add_trace(
#             go.Scatter(
#                 x=frame.x_grid,
#                 y=frame.element_outputs[i],
#                 mode="lines",
#                 name=f"Нейрон {i + 1}",
#                 line={"width": 1},
#                 opacity=0.4,
#                 showlegend=False,
#             )
#         )

#     fig.add_trace(
#         go.Scatter(
#             x=frame.x_grid,
#             y=frame.y_target,
#             mode="lines",
#             name="Целевая f(x)",
#             line={"color": "#FFFFFF", "width": 2.5, "dash": "dash"},
#         )
#     )

#     y_train_plot = np.where(frame.train_mask, frame.y_pred, np.nan)
#     y_val_plot = np.where(~frame.train_mask, frame.y_pred, np.nan)

#     fig.add_trace(
#         go.Scatter(
#             x=frame.x_grid,
#             y=y_train_plot,
#             mode="lines",
#             name="Предсказание (Train)",
#             line={"color": "#00D2FF", "width": 4},
#         )
#     )

#     if np.any(~frame.train_mask):
#         fig.add_trace(
#             go.Scatter(
#                 x=frame.x_grid,
#                 y=y_val_plot,
#                 mode="lines",
#                 name="Экстраполяция (Val)",
#                 line={"color": "#FF4B4B", "width": 3, "dash": "dot"},
#             )
#         )

#     fig.update_layout(
#         title={"text": f"Эпоха: {frame.epoch}", "font": {"size": 20, "color": "white"}},
#         xaxis_title="x",
#         yaxis_title="y",
#         yaxis={"range": y_range},  # Зажимаем Y в адекватные рамки
#         hovermode="x unified",
#         height=620,
#         margin={"l": 20, "r": 20, "t": 50, "b": 20},
#         legend={
#             "yanchor": "top",
#             "y": 0.99,
#             "xanchor": "right",
#             "x": 0.99,
#             "bgcolor": "rgba(0,0,0,0.5)",
#         },
#     )

#     return fig
