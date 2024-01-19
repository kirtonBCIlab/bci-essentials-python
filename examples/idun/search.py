import asyncio
from idun_guardian_client import GuardianClient

bci = GuardianClient()

# start a recording session
device_address = asyncio.run(bci.search_device())