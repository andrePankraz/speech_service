class RecorderProcessor extends AudioWorkletProcessor {

  constructor() {
    super()
    this.bufferSize = 16000 * 3 // 16 kHz with 3 seconds
    this._buffer = new Float32Array(this.bufferSize)
    this._bytesWritten = 0
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
      if (i % 3 !== 0) {
        // simple conversion from 48 kHz to 16 kHz by skipping 2 samples,
        // reduce sampling rate here in worker to reduce data transmissions
        continue
      }
      for (let j = 0; j < channels; ++j) {
        sample += input[j][i]
      }
      this._buffer[this._bytesWritten++] = sample / samples
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
    // 128 samples at 48 kHz is is 2.66 ms
    this.append(inputs[0])
    return true // keep processor alive
  }

}

registerProcessor("recorder.worklet", RecorderProcessor)
