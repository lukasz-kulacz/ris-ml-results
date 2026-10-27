import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd

    return Path, mo, np, pd


@app.cell
def _(Path, pd):
    DATA_DIR = Path(__file__).parent / "data_26_10_2026"

    def read_measurement(path):
        # Próbki z fazy settling są pomijane już przy wczytywaniu
        df = pd.read_csv(path, parse_dates=["Timestamp"])
        return df[~df["Settling"]].drop(columns="Settling")

    # Pomiar szumu (generator wyłączony) — osobny plik noise_*
    noise = pd.concat(
        [read_measurement(path) for path in sorted(DATA_DIR.glob("noise_*.csv"))],
        ignore_index=True,
    )

    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.stem.startswith("noise_"):
            continue
        # np. "random_choice_hold_toggle_-38dBm" -> algorytm, tryb, moc
        algorithm, mode, power = path.stem.rsplit("_", 2)
        df = read_measurement(path)
        df["algorithm"] = algorithm
        df["mode"] = mode.replace("continous", "continuous")  # literówka w pliku best_search
        df["power_dbm"] = int(power.removesuffix("dBm"))
        if algorithm != 'noise':
            df["Result"] = df["Result"] - 0.0
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data, noise


@app.cell
def _(data):
    # Kontrola: każda kombinacja algorytm/tryb powinna mieć tyle samo wierszy
    data.groupby(["algorithm", "mode"]).size().rename("rows").reset_index()
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data):
    import plotly.graph_objects as go

    # Stała kolejność i kolory algorytmów — ten sam algorytm ma ten sam kolor na obu wykresach
    ALGORITHM_COLORS = {
        "none": "#000000",
        "best_search": "#2a78d6",
        "mab": "#eb6834",
        "random_choice": "#1baf7a",
        "random_choice_hold": "#eda100",
    }

    # Oś czasu: kolejny numer próbki w obrębie pliku (Iteration zeruje się w każdym punkcie)
    samples = data.sort_values(["algorithm", "mode", "Point", "Iteration"]).copy()
    samples["sample"] = samples.groupby(["algorithm", "mode"]).cumcount() + 1

    def median_filtered(df, window):
        # Mediana krocząca (wyśrodkowana) liczona osobno dla każdego algorytmu i punktu
        if window <= 1:
            return df["Result"]
        return df.groupby(["algorithm", "Point"])["Result"].transform(
            lambda r: r.rolling(window, center=True, min_periods=1).median()
        )

    def power_over_time(mode, window=1):
        df = samples[samples["mode"] == mode].copy()
        df["filtered"] = median_filtered(df, window)
        fig = go.Figure()

        # Granice punktów pomiarowych (struktura identyczna dla wszystkich algorytmów)
        points = df.groupby("Point")["sample"].agg(["min", "max"])
        for point, p in points.iterrows():
            if p["min"] > 1:
                fig.add_vline(x=p["min"] - 0.5, line_width=1, line_dash="dot", line_color="#8a8880")
            fig.add_annotation(
                x=(p["min"] + p["max"]) / 2, y=1, yref="paper", yanchor="bottom",
                text=f"Punkt {point}", showarrow=False, font=dict(size=11, color="#6b6a64"),
            )

        for algorithm, color in ALGORITHM_COLORS.items():
            a = df[df["algorithm"] == algorithm]
            if a.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=a["sample"], y=a["filtered"],
                mode="lines", name=algorithm, legendgroup=algorithm,
                line=dict(color=color, width=1.5),
                customdata=a[["Point", "Iteration"]],
                hovertemplate="%{y:.2f} dBm<br>punkt %{customdata[0]}, iteracja %{customdata[1]}",
            ))

        filter_label = f", mediana z {window} próbek" if window > 1 else ""
        fig.update_layout(
            title=f"Moc odebrana w czasie — tryb {mode}{filter_label}",
            xaxis_title="Numer próbki",
            yaxis_title="Moc [dBm]",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            margin=dict(t=90),
            legend=dict(orientation="h", y=-0.15),
        )
        return fig

    return ALGORITHM_COLORS, go, power_over_time, samples


@app.cell
def _(mo):
    median_window = mo.ui.slider(
        start=1, stop=100, value=1, step=1, show_value=True,
        label="Filtr medianowy — okno [próbki] (1 = bez filtru)",
    )
    median_window
    return (median_window,)


@app.cell
def _():
    # power_over_time("toggle", median_window.value)
    return


@app.cell
def _(median_window, power_over_time):
    power_over_time("continuous", median_window.value)
    return


@app.cell
def _(ALGORITHM_COLORS, go, np, samples):
    from plotly.subplots import make_subplots

    def power_cdf(mode):
        # Tylko próbki z włączonym generatorem, surowe — bez filtru medianowego.
        # W trybie toggle odrzuca to połowę próbek (sam szum); w continuous generator jest zawsze włączony
        df = samples[
            (samples["mode"] == mode) & samples["GeneratorEnabled"]
        ]
        point_ids = sorted(df["Point"].unique())

        # Na górze zbiorczy CDF na całą szerokość, pod nim po jednym panelu na punkt
        cols = 3
        point_rows = -(-len(point_ids) // cols)
        fig = make_subplots(
            rows=1 + point_rows, cols=cols,
            specs=[[{"colspan": cols}] + [None] * (cols - 1)] + [[{}] * cols] * point_rows,
            subplot_titles=["Wszystkie punkty"] + [f"Punkt {pt}" for pt in point_ids],
            row_heights=[1.4] + [1] * point_rows,
            vertical_spacing=0.08, horizontal_spacing=0.05,
        )
        panels = [(None, 1, 1)] + [
            (pt, 2 + i // cols, 1 + i % cols) for i, pt in enumerate(point_ids)
        ]

        for algorithm, color in ALGORITHM_COLORS.items():
            a = df[df["algorithm"] == algorithm]
            if a.empty:
                continue
            for pt, row, col in panels:
                values = np.sort((a if pt is None else a[a["Point"] == pt])["Result"].to_numpy())
                fig.add_trace(go.Scatter(
                    x=values, y=np.arange(1, len(values) + 1) / len(values),
                    mode="lines", line=dict(color=color, width=1.5, shape="hv"),
                    name=algorithm, legendgroup=algorithm, showlegend=pt is None,
                    hovertemplate=f"{algorithm}<br>%{{x:.2f}} dBm<br>P = %{{y:.3f}}<extra></extra>",
                ), row=row, col=col)

        # Wspólna oś X we wszystkich panelach, żeby punkty dało się porównać.
        # Ucięta na 99,5 percentylu — pojedyncze skoki nie spłaszczają wykresu (zoom nadal działa)
        x_max = df["Result"].quantile(0.995)
        fig.update_xaxes(range=[df["Result"].min() - 0.2, x_max + 0.3])
        fig.update_yaxes(range=[0, 1.02], tickformat=".1f")
        fig.update_yaxes(title_text="CDF", row=1, col=1)
        for r in range(2, 2 + point_rows):
            fig.update_yaxes(title_text="CDF", row=r, col=1)
        fig.update_xaxes(title_text="Moc [dBm]", row=1 + point_rows)
        fig.update_layout(
            title=f"CDF mocy odebranej (generator włączony) — tryb {mode}",
            template="plotly_white",
            height=320 + 280 * point_rows,
            legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
        )
        return fig

    return make_subplots, power_cdf


@app.cell
def _():
    # power_cdf("toggle")
    return


@app.cell
def _(power_cdf):
    power_cdf("continuous")
    return


@app.cell
def _(mo, noise, np):
    # Moc szumu wyłącznie z pliku noise_* (generator wyłączony).
    # Uśrednianie w skali liniowej (mW), wynik z powrotem w dBm
    noise_samples = noise["Result"]
    noise_power_dbm = 10 * np.log10(np.mean(10 ** (noise_samples / 10)))

    mo.md(
        f"""
        **Średnia moc szumu:** {noise_power_dbm:.3f} dBm
        (z {len(noise_samples)} próbek z pomiaru szumu;
        odchylenie std. {noise_samples.std():.3f} dB)
        """
    )
    return (noise_power_dbm,)


@app.cell
def _(mo):
    detection_margin = mo.ui.number(
        start=0, stop=10, step=0.01, value=0.05,
        label="Próg detekcji — margines ponad szum [dB]",
    )
    detection_margin
    return (detection_margin,)


@app.cell
def _(detection_margin, noise_power_dbm, pd, samples):
    detection_threshold_dbm = noise_power_dbm + detection_margin.value

    def detection_rates(df):
        # P_d — odsetek próbek z włączonym generatorem powyżej progu (skuteczność detekcji)
        # P_fa — odsetek próbek z wyłączonym generatorem powyżej progu (fałszywe alarmy, tylko toggle)
        detected = df["Result"] > detection_threshold_dbm
        on = df["GeneratorEnabled"]
        return pd.Series({
            "P_d [%]": 100 * detected[on].mean(),
            "P_fa [%]": 100 * detected[~on].mean() if (~on).any() else float("nan"),
            "próbki z sygnałem": int(on.sum()),
        })

    detection = pd.concat([
        samples.groupby(["mode", "algorithm"]).apply(detection_rates, include_groups=False)
            .assign(point="Wszystkie"),
        samples.groupby(["mode", "algorithm", "Point"]).apply(detection_rates, include_groups=False)
            .reset_index("Point").assign(point=lambda d: "Punkt " + d["Point"].astype(str))
            .drop(columns="Point"),
    ]).reset_index()
    return detection, detection_threshold_dbm


@app.cell
def _(ALGORITHM_COLORS, detection, detection_threshold_dbm, go, make_subplots):
    def detection_chart(mode):
        df = detection[detection["mode"] == mode]
        metrics = ["P_d [%]"] + (["P_fa [%]"] if df["P_fa [%]"].notna().any() else [])
        titles = {
            "P_d [%]": "Skuteczność detekcji P_d (generator włączony)",
            "P_fa [%]": "Fałszywe alarmy P_fa (generator wyłączony)",
        }
        fig = make_subplots(
            rows=len(metrics), cols=1, shared_xaxes=True, vertical_spacing=0.12,
            subplot_titles=[titles[m] for m in metrics],
        )
        for row, metric in enumerate(metrics, start=1):
            for algorithm, color in ALGORITHM_COLORS.items():
                a = df[df["algorithm"] == algorithm]
                if a.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=a["point"], y=a[metric], name=algorithm, legendgroup=algorithm,
                    showlegend=row == 1, marker=dict(color=color, line=dict(width=0)),
                    hovertemplate=f"{algorithm}<br>%{{x}}: %{{y:.1f}}%<extra></extra>",
                ), row=row, col=1)
            fig.update_yaxes(title_text=metric, range=[0, 100], row=row, col=1)
            # Oddzielenie wyniku zbiorczego od punktów
            fig.add_vline(x=0.5, line_width=1, line_dash="dot", line_color="#8a8880", row=row, col=1)

        fig.update_layout(
            title=f"Detekcja sygnału (próg {detection_threshold_dbm:.3f} dBm) — tryb {mode}",
            template="plotly_white",
            barmode="group", bargap=0.25, bargroupgap=0.05,
            height=300 + 260 * len(metrics),
            legend=dict(orientation="h", yanchor="top", y=-0.08, x=0),
        )
        return fig

    return (detection_chart,)


@app.cell
def _():
    # detection_chart("toggle")
    return


@app.cell
def _(detection_chart):
    detection_chart("continuous")
    return


@app.cell
def _(detection):
    # Tabela z liczbami do wykresów detekcji
    detection.pivot_table(
        index=["mode", "point"], columns="algorithm", values="P_d [%]"
    ).round(1)
    return


if __name__ == "__main__":
    app.run()
