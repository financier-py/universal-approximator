import streamlit as st

from src.data_builder import DataBuilder
from src.graphics import create_animated_figure, export_frames_to_gif
from src.parser import ExpressionParser, InvalidExpressionError
from src.schemas.experiment_params import ExperimentParams
from src.schemas.training_state import EpochState
from nn_trainer import Trainer

# НАСТРОЙКИ СТРАНИЦЫ
st.set_page_config(
    page_title="Аппроксиматор функций нейросетью",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Интерактивная аппроксимация функций")
st.caption(
    "Визуализация работы скрытого слоя нейросети: "
    "как отдельные нейроны собирают итоговый график функции."
)

# БОКОВАЯ ПАНЕЛЬ
with st.sidebar:
    st.header("⚙️ Параметры эксперимента")

    # формула + сетка
    expr_input = st.text_input(
        "Функция f(x):",
        value="sin(x) * exp(-0.1 * x)",
        help="Разрешена переменная x и функции: sin, cos, exp, log, abs, sqrt и куча других",
    )
    n_points_input = st.slider(
        "Количество точек сетки (n_points):",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
    )

    st.divider()

    # диапозоны
    st.subheader("📐 Границы диапазонов")
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        x_vis_min = st.number_input("x_vis_min", value=-5.0, step=0.5)
    with col_vis2:
        x_vis_max = st.number_input("x_vis_max", value=5.0, step=0.5)

    col_tr1, col_tr2 = st.columns(2)
    with col_tr1:
        x_train_min = st.number_input("x_train_min", value=-2.0, step=0.5)
    with col_tr2:
        x_train_max = st.number_input("x_train_max", value=2.0, step=0.5)

    st.divider()

    # архитектура нейросети
    st.subheader("🧠 Нейросеть")
    n_neurons = st.slider("Количество нейронов:", min_value=1, max_value=50, value=10)
    activation = st.selectbox(
        "Активация:",
        options=["sigmoid", "relu", "tanh", "gelu", "leaky_relu"],
        index=0,
    )

    st.divider()

    # обучение
    st.subheader("🎓 Обучение")
    lr = st.number_input("Learning Rate:", value=0.01, format="%.4f", step=0.001)
    epochs = st.number_input("Количество эпох:", value=500, step=50, min_value=10)
    yield_every = st.number_input("Кадр каждые N эпох:", value=10, step=1, min_value=1)

    # скорость визуализации
    frame_duration = st.slider(
        "Длительность кадра (мс):",
        min_value=10,
        max_value=200,
        value=40,
        step=10,
        help="Скорость воспроизведения анимации",
    )

    start_button = st.button("🚀 Запустить обучение", type="primary", width="stretch")


# логика приложения
if start_button:
    try:
        params = ExperimentParams(
            expression=expr_input,
            n_points=n_points_input,
            x_vis_min=x_vis_min,
            x_vis_max=x_vis_max,
            x_train_min=x_train_min,
            x_train_max=x_train_max,
            n_neurons=n_neurons,
            activation_type=activation,  # type: ignore
            learning_rate=lr,
            epochs=int(epochs),
            yield_every=int(yield_every),
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Ошибка в переданных параметрах эксперимента: {e}")
        st.stop()

    # компилим ф-ию
    try:
        safe_func = ExpressionParser.parse(params.expression)
    except InvalidExpressionError as e:
        st.error(f"❌ Ошибка в ф-ии переданной: {e}")
        st.stop()

    # делаем датасет
    try:
        prepared_data = DataBuilder.create(params, safe_func)
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Ошибка подготовки данных: {e}")
        st.stop()

    # Обучение
    st.info("⌛ Обучение нейросети и рендеринг кадров...")
    progress_bar = st.progress(0)
    trainer = Trainer(params, prepared_data)

    all_frames: list[EpochState] = []
    for frame in trainer.train():
        all_frames.append(frame)
        progress_bar.progress(frame.epoch / params.epochs)

    progress_bar.empty()

    st.session_state["all_frames"] = all_frames
    st.session_state["params"] = params


if "all_frames" in st.session_state:
    all_frames = st.session_state["all_frames"]
    params = st.session_state["params"]

    last_frame = all_frames[-1]

    col1, col2 = st.columns(2)
    col1.metric("RMSE Train", f"{last_frame.loss_train:.5f}")
    col2.metric("RMSE Val", f"{last_frame.loss_val:.5f}")

    fig = create_animated_figure(all_frames, frame_duration)
    st.plotly_chart(fig, width="stretch")

    with st.expander("📥 Скачать анимацию в формате GIF"):
        if st.button("Сгенерировать GIF"):
            with st.spinner("Создаём GIF-файл..."):
                gif_bytes = export_frames_to_gif(all_frames, fps=20)
                st.session_state["gif_bytes"] = gif_bytes

        if "gif_bytes" in st.session_state:
            st.download_button(
                label="💾 Сохранить .gif",
                data=st.session_state["gif_bytes"],
                file_name=f"approximator_{params.expression}.gif",
                mime="image/gif",
                type="primary",
            )

else:
    st.info(
        "👈 Настройте параметры на боковой панели и нажмите **'Запустить обучение'**."
    )
