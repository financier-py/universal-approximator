import streamlit as st

from src.data_builder import DataBuilder
from src.graphics import create_animated_figure, export_frames_to_gif
from src.parser import ExpressionParser, InvalidExpressionError
from src.schemas.experiment_params import ExperimentParams
from src.schemas.training_state import EpochState
from src.trainer_factory import get_trainer

# НАСТРОЙКИ СТРАНИЦЫ
st.set_page_config(
    page_title="Аппроксиматор функций",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Интерактивная аппроксимация функций")
st.caption(
    "Визуализация работы нейросети или же град. бустинга: "
    "как отдельные нейроны или деревья собирают итоговый график функции."
)

if "model_choice_val" not in st.session_state:
    st.session_state["model_choice_val"] = "Нейросеть (MLP)"

# БОКОВАЯ ПАНЕЛЬ
with st.sidebar:
    st.header("⚙️ Параметры эксперимента")

    # Выбор модели
    model_choice = st.radio(
        "Алгоритм:",
        options=["Нейросеть (MLP)", "Градиентный бустинг"],
        key="model_choice_val",
    )
    model_type = "gdbt" if model_choice == "Градиентный бустинг" else "nn"

    st.divider()

    # формула + сетка
    expr_input = st.text_input(
        "Функция f(x):",
        value="sin(x) * exp(-0.1 * x)",
        help="Разрешена переменная x и функции: sin, cos, exp, log, abs, sqrt и другие",
    )
    n_points_input = st.slider(
        "Количество точек сетки (n_points):",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
    )

    st.divider()

    # диапазоны
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

    # специфичные настройки для выбранной архитектуры
    if model_type == "nn":
        st.subheader("Нейросеть")
        n_neurons = st.slider(
            "Количество нейронов:", min_value=1, max_value=50, value=10
        )
        activation = st.selectbox(
            "Активация:",
            options=["sigmoid", "relu", "tanh", "gelu", "leaky_relu"],
            index=0,
        )
        epochs = st.number_input("Количество эпох:", value=500, step=50, min_value=10)
        # заглушки для параметров gdbt
        n_estimators = 100
        max_depth = 3
        gdbt_view_mode = "current"
    else:
        st.subheader("Градиентный бустинг")
        n_estimators = st.number_input(
            "Количество деревьев (n_estimators):", value=100, step=10, min_value=5
        )
        max_depth = st.slider(
            "Макс. глубина дерева (max_depth):", min_value=1, max_value=10, value=3
        )
        gdbt_view_mode_choice = st.selectbox(
            "Отображение деревьев:",
            options=[
                "Текущее дерево (без сжатия)",
                "История прошлых деревьев",
            ],
            index=0,
        )
        gdbt_view_mode = (
            "current"
            if gdbt_view_mode_choice == "Текущее дерево (без сжатия)"
            else "history"
        )
        epochs = int(n_estimators)
        # заглушки для параметров NN
        n_neurons = 10
        activation = "sigmoid"

    st.divider()

    # общие параметры обучения
    st.subheader("🎓 Обучение")
    lr = st.number_input(
        "Learning Rate:",
        value=0.01 if model_type == "nn" else 0.1,
        format="%.4f",
        step=0.001,
    )
    yield_every = st.number_input("Кадр каждые N шагов:", value=10, step=1, min_value=1)

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
    if "all_frames" in st.session_state:
        del st.session_state["all_frames"]
    if "gif_bytes" in st.session_state:
        del st.session_state["gif_bytes"]
    try:
        params = ExperimentParams(
            model_type=model_type,  # type: ignore
            expression=expr_input,
            n_points=n_points_input,
            x_vis_min=x_vis_min,
            x_vis_max=x_vis_max,
            x_train_min=x_train_min,
            x_train_max=x_train_max,
            n_neurons=n_neurons,
            activation_type=activation,  # type: ignore
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            gdbt_view_mode=gdbt_view_mode,  # type: ignore
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

    # обучение
    st.info("⌛ Обучение нейросети и рендеринг кадров...")
    progress_bar = st.progress(0)

    # получаем нужный трейнер через фабрику
    trainer = get_trainer(params, prepared_data)

    all_frames: list[EpochState] = []
    total_steps = params.epochs if model_type == "nn" else params.n_estimators

    for frame in trainer.train():
        all_frames.append(frame)
        progress_bar.progress(min(frame.epoch / total_steps, 1.0))
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
