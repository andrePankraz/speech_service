# speech_service
Some tests with AI speech models as web services

## Start as local Service
- Clone https://github.com/andrePankraz/speech_service
- docker compose up
  - Will take some time, downloading nearly 20 GB the first time
  - Wait & check in up and running
- Go to URL: http://localhost:8200/
  - Models are downloaded the first time, will take some time, multiple GB

## Develop
- Clone https://github.com/andrePankraz/speech_service
- Uncomment TARGET=dev in .env
- docker compose up
  - Will take some time, downloading nearly 20 GB the first time
  - Wait & check in up and running
- Install VS Code
  - Install Extension
    - Docker
    - Dev Containers
    - Markdown Preview
- VS Code:
  - Attach to running containers... (Lower left edge in VS Code)
    - select speech_service-paython
  - Explorer Open folder -> /opt/speech_service
  - Run / Start Debug
    - VS Code Extension Python will be installed the first time (Wait and another Start Debug)
    - Select Python Interpreter
- Go to URL: http://localhost:8200/
  - Models are downloaded the first time, will take some time, multiple GB
