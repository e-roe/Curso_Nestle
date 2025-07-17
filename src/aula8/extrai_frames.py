import cv2
import os
import argparse


def extrai_frames(video_path, output_dir, start_time=0, duration=None):
    """
    Extrai frames de um vídeo e os salva em um diretório especificado.

    :param video_path: Caminho para o arquivo de vídeo de entrada.
    :param output_dir: Diretório onde os frames serão salvos.
    :param start_time: Tempo inicial em segundos (padrão é 0).
    :param duration: Duração em segundos para extrair os frames (padrão é None, ou seja, até o final).
    """
    # Garante que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    # Abre o arquivo de vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o arquivo de vídeo {video_path}")
        return

    # Obtém as propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    # Calcula o intervalo de frames
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps) if duration else total_frames
    end_frame = min(end_frame, total_frames)

    # Define o frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    saved_count = 0

    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Salva o frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extração concluída. {saved_count} frames salvos em {output_dir}.")


# Uso
# python extrai_frames.py path/video1.mp4 path/frames1 --start_time 2 --duration 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai frames de um vídeo.")
    parser.add_argument("video_path", type=str, help="Caminho para o arquivo de vídeo de entrada.")
    parser.add_argument("output_dir", type=str, help="Diretório onde os frames serão salvos.")
    parser.add_argument("--start_time", type=float, default=0, help="Tempo inicial em segundos (padrão: 0).")
    parser.add_argument("--duration", type=float, default=None, help="Duração em segundos para extrair os frames (padrão: até o final).")

    args = parser.parse_args()

    extrai_frames(args.video_path, args.output_dir, args.start_time, args.duration)