class RecorderProcessor extends AudioWorkletProcessor {

  constructor() {
    super()
    this.bufferSize = 16000 * 5 // 16 kHz with 2 seconds
    this._bytesWritten = 0
    this._buffer = new Float32Array(this.bufferSize)
    this.initBuffer()
  }

  initBuffer() {
    this._bytesWritten = 0
  }

  isBufferEmpty() {
    return this._bytesWritten === 0
  }

  isBufferFull() {
    return this._bytesWritten === this.bufferSize
  }

  append(input) {
    const channels = input.length
    const samples = input[0].length
    let sample = 0
    for (let i = 0; i < samples; i++) {
      for (let j = 0; j < channels; ++j) {
        sample += input[j][i]
      }
      if (i % 3 !== 0) {
        // simple 48000 -> 16000 kHz bei skipping, later interpolation?
        // reduce here in worker to reduce data transmissions
        continue
      }
      this._buffer[this._bytesWritten++] = sample / (samples * 3)
      sample = 0
      if (this.isBufferFull()) {
        this.flush()
      }
    }
  }

  flush() {
    // trim the buffer if ended prematurely
    this.port.postMessage(
      this._bytesWritten < this.bufferSize
        ? this._buffer.slice(0, this._bytesWritten)
        : this._buffer
    )
    this.initBuffer()
  }

  process(inputs, outputs) {
    // input[n'th input][m'th channel] -> Float32Array with 128 sample values in range [-1 .. 1]
    this.append(inputs[0])
    return true // keep processor alive
  }

}

registerProcessor("recorder.worklet", RecorderProcessor)
