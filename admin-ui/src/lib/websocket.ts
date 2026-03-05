type MessageHandler = (data: unknown) => void;
type StateHandler   = (state: ConnectionState) => void;

export type ConnectionState = 'connecting' | 'connected' | 'disconnected';

export class WebSocketManager {
  private url:             string;
  private onMessage:       MessageHandler;
  private onStateChange?:  StateHandler;
  private ws:              WebSocket | null = null;
  private reconnectTimer:  ReturnType<typeof setTimeout> | null = null;
  private backoff:         number = 1000;
  private readonly maxBackoff = 30_000;
  private destroyed        = false;
  private _state:          ConnectionState = 'disconnected';

  constructor(url: string, onMessage: MessageHandler, onStateChange?: StateHandler) {
    this.url           = url;
    this.onMessage     = onMessage;
    this.onStateChange = onStateChange;
  }

  connect(): void {
    if (this.destroyed || this.ws) return;
    this._setState('connecting');
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.backoff = 1000;
        this._setState('connected');
      };

      this.ws.onmessage = (ev) => {
        try {
          this.onMessage(JSON.parse(ev.data));
        } catch {
          this.onMessage(ev.data);
        }
      };

      this.ws.onerror = () => { /* handled by onclose */ };

      this.ws.onclose = () => {
        this.ws = null;
        if (!this.destroyed) {
          this._setState('disconnected');
          this._scheduleReconnect();
        }
      };
    } catch {
      this.ws = null;
      this._setState('disconnected');
      this._scheduleReconnect();
    }
  }

  disconnect(): void {
    this.destroyed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this._setState('disconnected');
  }

  get state(): ConnectionState {
    return this._state;
  }

  private _setState(s: ConnectionState): void {
    this._state = s;
    this.onStateChange?.(s);
  }

  private _scheduleReconnect(): void {
    if (this.destroyed) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.destroyed) this.connect();
    }, this.backoff);
    this.backoff = Math.min(this.backoff * 2, this.maxBackoff);
  }
}
