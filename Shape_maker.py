
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from shapely.geometry import shape, LineString


class FourierCountryDrawer:
    """Draw a country outline using Fourier coefficients and epicycles."""

    GEOJSON_URL = (
        "https://raw.githubusercontent.com/"
        "datasets/geo-countries/master/data/countries.geojson"
    )

    def __init__(
        self,
        country="Tunisia",
        n_samples=1024,
        n_terms=100,
        cache_file="countries.geojson",
    ):
        if n_samples < 16:
            raise ValueError("n_samples must be at least 16.")

        if n_terms < 1:
            raise ValueError("n_terms must be at least 1.")

        self.country = country
        self.n_samples = n_samples
        self.n_terms = n_terms

        # Cache file is stored next to this Python script.
        self.cache_path = Path(__file__).resolve().parent / cache_file

        self.xy = None
        self.z = None
        self.coefficients = None
        self.frequencies = None
        self.amplitudes = None
        self.phases = None
        self.z_reconstructed = None

    # --------------------------------------------------------
    # 1. DOWNLOAD AND CACHE COUNTRY DATA
    # --------------------------------------------------------
    def _get_geojson(self, retries=3, timeout=120):
        # Use the local copy if it exists.
        if self.cache_path.exists():
            print(f"Loading cached data: {self.cache_path}")

            try:
                with self.cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as error:
                print(f"Cache could not be read: {error}")
                print("Downloading a fresh copy...")

        for attempt in range(1, retries + 1):
            try:
                print(
                    f"Downloading country data "
                    f"(attempt {attempt}/{retries})..."
                )

                request = urllib.request.Request(
                    self.GEOJSON_URL,
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                with urllib.request.urlopen(
                    request, timeout=timeout
                ) as response:
                    data = json.load(response)

                # Save only after a successful download and parse.
                try:
                    with self.cache_path.open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(data, f)
                    print("Country data cached successfully.")
                except OSError as error:
                    print(f"Could not save cache: {error}")

                return data

            except (
                TimeoutError,
                urllib.error.URLError,
                json.JSONDecodeError,
                OSError,
            ) as error:
                print(f"Download failed: {error}")

                if attempt < retries:
                    time.sleep(2 * attempt)

        raise RuntimeError(
            "Could not download country data after several attempts. "
            "Check your internet connection and try again."
        )

    # --------------------------------------------------------
    # 2. LOAD AND PROCESS THE COUNTRY BORDER
    # --------------------------------------------------------
    def load_country(self):
        geojson = self._get_geojson()

        # Find the requested country by its property values.
        feature = next(
            (
                f for f in geojson["features"]
                if any(
                    str(value).strip().casefold()
                    == self.country.strip().casefold()
                    for value in f["properties"].values()
                    if value is not None
                )
            ),
            None,
        )

        if feature is None:
            raise ValueError(
                f"Country '{self.country}' was not found in the dataset."
            )

        geometry = shape(feature["geometry"])

        # Use the largest polygon (mainland).
        if geometry.geom_type == "MultiPolygon":
            polygon = max(geometry.geoms, key=lambda p: p.area)
        elif geometry.geom_type == "Polygon":
            polygon = geometry
        else:
            raise ValueError(
                f"Unsupported geometry type: {geometry.geom_type}"
            )

        lon, lat = np.asarray(polygon.exterior.coords).T

        # Simple longitude correction for a visually useful projection.
        mean_lat = np.deg2rad(lat.mean())

        xy = np.column_stack([
            lon * np.cos(mean_lat),
            lat,
        ])

        # Resample uniformly by distance along the outline.
        border = LineString(xy)

        points = [
            border.interpolate(
                j / self.n_samples, normalized=True
            )
            for j in range(self.n_samples)
        ]

        self.xy = np.array([(p.x, p.y) for p in points])

        # Center and scale the outline.
        self.xy -= self.xy.mean(axis=0)
        scale = np.max(np.abs(self.xy))

        if scale == 0:
            raise ValueError("The country outline has zero size.")

        self.xy /= scale

        # Represent 2D coordinates as one complex signal.
        self.z = self.xy[:, 0] + 1j * self.xy[:, 1]

        print(f"\nLoaded country: {self.country}")
        print(f"Border samples: {len(self.z)}")

        return self

    # --------------------------------------------------------
    # 3. COMPUTE THE FOURIER COEFFICIENTS
    # --------------------------------------------------------
    def compute_fourier(self):
        if self.z is None:
            raise RuntimeError("Call load_country() first.")

        N = len(self.z)

        # Normalized DFT.
        C_all = np.fft.fft(self.z) / N

        # Signed frequencies corresponding to FFT coefficients.
        k_all = np.rint(
            np.fft.fftfreq(N, d=1 / N)
        ).astype(int)

        K = min(self.n_terms, N)

        # Keep the K largest coefficients by magnitude.
        indices = np.argsort(np.abs(C_all))[-K:]

        # Largest circle first for the epicycle animation.
        indices = indices[
            np.argsort(-np.abs(C_all[indices]))
        ]

        self.coefficients = C_all[indices]
        self.frequencies = k_all[indices]
        self.amplitudes = np.abs(self.coefficients)
        self.phases = np.angle(self.coefficients)

        print(f"Selected Fourier components: {K}")

        return self

    # --------------------------------------------------------
    # 4. RECONSTRUCT THE COUNTRY OUTLINE
    # --------------------------------------------------------
    def reconstruct(self):
        if self.coefficients is None:
            raise RuntimeError("Call compute_fourier() first.")

        t = np.arange(self.n_samples) / self.n_samples

        self.z_reconstructed = np.sum(
            self.coefficients[:, None]
            * np.exp(
                2j * np.pi
                * self.frequencies[:, None]
                * t[None, :]
            ),
            axis=0,
        )

        rmse = np.sqrt(
            np.mean(
                np.abs(self.z - self.z_reconstructed) ** 2
            )
        )

        print(f"Reconstruction RMSE: {rmse:.6f}")

        return self

    # --------------------------------------------------------
    # 5. PLOT OUTLINE, AMPLITUDE AND PHASE
    # --------------------------------------------------------
    def plot_results(self):
        if self.z_reconstructed is None:
            raise RuntimeError("Call reconstruct() first.")

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Original and reconstructed outline.
        axes[0].plot(
            self.z.real, self.z.imag,
            "--", label="Original"
        )
        axes[0].plot(
            self.z_reconstructed.real,
            self.z_reconstructed.imag,
            label=f"Fourier ({len(self.coefficients)} terms)"
        )
        axes[0].set_title(f"{self.country}: reconstruction")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].grid(True)
        axes[0].legend()

        # Complete amplitude spectrum.
        C_all = np.fft.fft(self.z) / len(self.z)
        k_all = np.rint(
            np.fft.fftfreq(len(self.z), d=1 / len(self.z))
        ).astype(int)

        axes[1].stem(k_all, np.abs(C_all))
        axes[1].set_title("Fourier amplitude spectrum")
        axes[1].set_xlabel("Signed frequency k")
        axes[1].set_ylabel("|C_k|")
        axes[1].grid(True)

        # Phase of the selected coefficients.
        axes[2].scatter(self.frequencies, self.phases)
        axes[2].set_title("Selected Fourier phases")
        axes[2].set_xlabel("Signed frequency k")
        axes[2].set_ylabel("Phase (radians)")
        axes[2].grid(True)

        fig.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 6. ANIMATE FOURIER EPICYCLES
    # --------------------------------------------------------
    def animate(self, frames=240, interval=30):
        if self.coefficients is None:
            raise RuntimeError("Call compute_fourier() first.")

        C = self.coefficients
        k = self.frequencies
        K = len(C)

        times = np.linspace(0, 1, frames, endpoint=False)

        # Rotating vector for each coefficient at each frame.
        vectors = (
            C[:, None]
            * np.exp(
                2j * np.pi * k[:, None] * times[None, :]
            )
        )

        # Cumulative endpoints of all rotating vectors.
        endpoints = np.vstack([
            np.zeros((1, frames), dtype=complex),
            np.cumsum(vectors, axis=0),
        ])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{self.country}: Fourier epicycles")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True)

        limit = 1.4
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        # Original outline for comparison.
        ax.plot(
            self.z.real, self.z.imag,
            "--", alpha=0.3, label="Original"
        )

        # Create artists once; update them for every frame.
        lines = [
            ax.plot([], [], linewidth=0.8)[0]
            for _ in range(K)
        ]

        circles = [
            Circle(
                (0, 0),
                radius=float(abs(c)),
                fill=False,
                linewidth=0.65,
                alpha=0.5,
            )
            for c in C
        ]

        for circle in circles:
            ax.add_patch(circle)

        trace, = ax.plot([], [], linewidth=2, label="Trace")
        endpoint, = ax.plot([], [], "ro", markersize=4)

        ax.legend()

        def update(frame):
            for j in range(K):
                start = endpoints[j, frame]
                end = endpoints[j + 1, frame]

                lines[j].set_data(
                    [start.real, end.real],
                    [start.imag, end.imag],
                )

                circles[j].center = (start.real, start.imag)

            # Endpoint of the final rotating vector.
            point = endpoints[-1, frame]
            endpoint.set_data([point.real], [point.imag])

            # Draw the complete trail so far.
            trace.set_data(
                endpoints[-1, :frame + 1].real,
                endpoints[-1, :frame + 1].imag,
            )

            return lines + circles + [trace, endpoint]

        animation = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=interval,
            blit=False,
            repeat=True,
        )

        # Keep a reference to the animation on this object.
        self.animation = animation

        # In a normal Python script, plt.show() opens the animation
        # in a Matplotlib window. No Jupyter display is needed.
        plt.show()

        return animation


# ============================================================
# 7. RUN THE PROJECT
# ============================================================

if __name__ == "__main__":
    drawer = FourierCountryDrawer(
        country="Tunisia",
        n_samples=1024,
        n_terms=100,
    )

    drawer.load_country()
    drawer.compute_fourier()
    drawer.reconstruct()

    drawer.plot_results()
    drawer.animate()
