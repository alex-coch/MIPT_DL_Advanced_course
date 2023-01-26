import asyncio
import datetime
import aiohttp
import aiofiles
from typing import List
import os

from clients.tg import TgClient
from clients.tg.dcs import UpdateObj


class Worker:
    def __init__(self, token: str, queue: asyncio.Queue, concurrent_workers: int):
        self.tg_client = TgClient(token)
        self.queue = queue
        self.concurrent_workers = concurrent_workers
        self._tasks: List[asyncio.Task] = []
        self.token = token

    async def handle_update(self, upd: UpdateObj):
        chat_id = upd.message.chat.id
        id = list(sorted(upd.message.photo, key=lambda x: x['file_size'], reverse=True))[0]['file_id']
        print("start:", datetime.datetime.now())
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://api.telegram.org/bot{self.token}/getFile?file_id={id}') as resp:
                file_path = await resp.json()
                file_path = file_path['result']['file_path']
                # print(file_path)

                file_ext = file_path[file_path.rfind('.'):]

            async with session.get(f'https://api.telegram.org/file/bot{self.token}/{file_path}') as resp2:
                # print(resp2.status)
                if resp2.status == 200:
                    file_name = f'{id}{file_ext}'
                    # print(f'tmp file: {file_name}')
                    f = await aiofiles.open(file_name, mode='wb')
                    await f.write(await resp2.read())
                    await f.close()

            import cv2
            import numpy as np
            import torch
            import RRDBNet_arch as arch

            model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


            model = arch.RRDBNet(3, 3, 64, 23, gc=32)
            model.load_state_dict(torch.load(model_path), strict=True)
            model.eval()
            model = model.to(device)

            img = cv2.imread(file_name, cv2.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            with torch.no_grad():
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            cv2.imwrite(f'output-{file_name}', output)

            import telepot
            bot = telepot.Bot(self.token)
            bot.sendPhoto(chat_id, photo=open(f'output-{file_name}', 'rb'))

            os.remove(file_name)
            os.remove(f'output-{file_name}')

        print("end:", datetime.datetime.now())

    async def _worker(self):
        while True:
            upd = await self.queue.get()
            try:
                await self.handle_update(upd)
            finally:
                self.queue.task_done()

    async def start(self):
        self._tasks = [asyncio.create_task(self._worker()) for _ in range(self.concurrent_workers)]

    async def stop(self):
        await self.queue.join()
        for t in self._tasks:
            t.cancel()
