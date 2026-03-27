import h5py
import matplotlib.pyplot as plt

H5_PATH = "/home/teamb/Desktop/Sim2RealB_2/data/20250827_151212.h5"
DATASET = "observations/images/oakd_front_view/color"
START_FRAME = 0


def main():
    with h5py.File(H5_PATH, "r") as f:
        frames = f[DATASET]
        n_frames = frames.shape[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.12)

        idx = START_FRAME
        autoplay = False

        img = ax.imshow(frames[idx])
        title = ax.set_title(f"{DATASET} - frame {idx}/{n_frames-1}")
        ax.axis("off")

        def update_frame():
            img.set_data(frames[idx])
            title.set_text(f"{DATASET} - frame {idx}/{n_frames-1}")
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal idx, autoplay

            if event.key == "right":
                idx = min(idx + 1, n_frames - 1)
                update_frame()
            elif event.key == "left":
                idx = max(idx - 1, 0)
                update_frame()
            elif event.key == " ":
                autoplay = not autoplay
            elif event.key == "q":
                plt.close(fig)

        def autoplay_loop():
            nonlocal idx
            if autoplay:
                idx = min(idx + 1, n_frames - 1)
                update_frame()
                fig.canvas.new_timer(interval=30).add_callback(autoplay_loop).start()

        fig.canvas.mpl_connect("key_press_event", on_key)
        autoplay_loop()

        plt.show()


if __name__ == "__main__":
    main()