import asyncio
import datetime
import os
os.environ['PYTHONASYNCIODEBUG'] = '1'

from bot.base import Bot
from dotenv import load_dotenv

load_dotenv()

def run():
    # loop = asyncio.get_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_debug(True)
    bot = Bot(os.getenv("BOT_TOKEN"), 2)
    try:
        print('bot has been started')
        loop.create_task(bot.start())
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nstopping", datetime.datetime.now())
        loop.run_until_complete(bot.stop())
        print('bot has been stopped', datetime.datetime.now())


if __name__ == '__main__':
    run()
