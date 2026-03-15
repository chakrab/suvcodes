import json
import datetime

from agents import TracingProcessor
from agents.tracing import AgentSpanData, FunctionSpanData, GenerationSpanData, ResponseSpanData, HandoffSpanData, CustomSpanData, GuardrailSpanData, \
    TranscriptionSpanData, SpeechSpanData, SpeechGroupSpanData, MCPListToolsSpanData

class LocalTracingProcessor(TracingProcessor):
    """
    A tracing processor that builds a JSON flow. Traces contain spans (one trace -> many spans).
    On trace end the trace is serialized to JSON (printed or can be saved).
    """
    def __init__(self):
        # store trace dicts keyed by trace_id
        self.active_traces = {}
        # store raw span objects keyed by span_id until they end
        self.active_spans = {}

    def _make_trace_dict(self, trace):
        return {
            "trace_id": getattr(trace, "trace_id", None),
            "name": getattr(trace, "name", None),
            "workflow_name": getattr(trace, "workflow_name", None),
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "ended_at": None,
            "attributes": getattr(trace, "attributes", {}) or {},
            "spans": []
        }

    def _span_type_and_payload(self, span):
        sd = span.span_data
        payload = {}
        span_type = type(sd).__name__ if sd is not None else "Unknown"
        # extract useful fields per known span types
        if isinstance(sd, AgentSpanData):
            payload["name"] = getattr(sd, "name", None)
        elif isinstance(sd, FunctionSpanData):
            payload["name"] = getattr(sd, "name", None)
        elif isinstance(sd, GenerationSpanData):
            payload["model"] = getattr(sd, "model", None)
            payload["input"] = getattr(sd, "input", None)
            payload["output"] = getattr(sd, "output", None)
        elif isinstance(sd, ResponseSpanData):
            payload["response"] = getattr(sd, "Response", None)
        elif isinstance(sd, HandoffSpanData):
            payload["from_agent"] = getattr(sd, "from_agent", None)
            payload["to_agent"] = getattr(sd, "to_agent", None)
        elif isinstance(sd, CustomSpanData):
            payload["name"] = getattr(sd, "name", None)
        elif isinstance(sd, GuardrailSpanData):
            payload["name"] = getattr(sd, "name", None)
            payload["triggered"] = getattr(sd, "triggered", None)
        elif isinstance(sd, TranscriptionSpanData):
            payload["model"] = getattr(sd, "model", None)
        elif isinstance(sd, SpeechSpanData):
            payload["model"] = getattr(sd, "model", None)
        elif isinstance(sd, SpeechGroupSpanData):
            payload["input"] = getattr(sd, "input", None)
        elif isinstance(sd, MCPListToolsSpanData):
            payload["server"] = getattr(sd, "server", None)
        else:
            # generic fallback: try to convert attributes
            try:
                payload = sd.__dict__
            except Exception:
                payload = {}
        return span_type, payload

    def _get_span_trace_id(self, span):
        # Try common locations for trace id/pointer
        for attr in ("trace_id", "trace", "parent_trace_id", "traceId"):
            val = getattr(span, attr, None)
            if val:
                # if it's an object with trace_id
                if hasattr(val, "trace_id"):
                    return getattr(val, "trace_id")
                return val
        # fallback: if span has span_data with trace reference
        sd = getattr(span, "span_data", None)
        if sd:
            for attr in ("trace_id", "trace"):
                val = getattr(sd, attr, None)
                if val:
                    if hasattr(val, "trace_id"):
                        return getattr(val, "trace_id")
                    return val
        return None

    def on_trace_start(self, trace):
        tid = getattr(trace, "trace_id", None)
        if tid is None:
            # generate key if missing
            tid = id(trace)
        self.active_traces[tid] = self._make_trace_dict(trace)

    def on_trace_end(self, trace):
        tid = getattr(trace, "trace_id", None) or id(trace)
        tdict = self.active_traces.get(tid)
        if tdict is None:
            # create if missing
            tdict = self._make_trace_dict(trace)
            self.active_traces[tid] = tdict
        tdict["ended_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # serialize JSON flow for this trace
        print(json.dumps(tdict, indent=2, sort_keys=False, default=str), flush=True)
        # cleanup
        del self.active_traces[tid]

    def on_span_start(self, span):
        # store raw span until it ends
        sid = getattr(span, "span_id", None) or id(span)
        self.active_spans[sid] = span

    def on_span_end(self, span):
        sid = getattr(span, "span_id", None) or id(span)
        raw = self.active_spans.get(sid)
        if raw is None:
            return
        # build span dict
        span_type, payload = self._span_type_and_payload(raw)
        span_dict = {
            "span_id": getattr(raw, "span_id", None),
            "name": payload.get("name", getattr(raw, "name", None)),
            "type": span_type,
            "started_at": getattr(raw, "started_at", None),
            "ended_at": getattr(raw, "ended_at", None),
            "payload": payload,
            "attributes": getattr(raw, "attributes", {}) or {}
        }
        # attach to its trace
        trace_id = self._get_span_trace_id(raw) or getattr(raw, "trace_id", None) or getattr(raw, "trace", None)
        if hasattr(trace_id, "trace_id"):
            trace_id = getattr(trace_id, "trace_id")
        if trace_id is None:
            # if we can't find trace, put into a special orphan trace
            trace_id = "orphan"
            if trace_id not in self.active_traces:
                self.active_traces[trace_id] = {
                    "trace_id": trace_id,
                    "name": "orphan",
                    "started_at": None,
                    "ended_at": None,
                    "attributes": {},
                    "spans": []
                }
        if trace_id not in self.active_traces:
            # create minimal container
            self.active_traces[trace_id] = {
                "trace_id": trace_id,
                "name": None,
                "started_at": None,
                "ended_at": None,
                "attributes": {},
                "spans": []
            }
        self.active_traces[trace_id]["spans"].append(span_dict)
        # cleanup stored span
        del self.active_spans[sid]

    def shutdown(self):
        # Optionally flush remaining traces as JSON
        for tid, tdict in list(self.active_traces.items()):
            print(json.dumps(tdict, indent=2, sort_keys=False, default=str), flush=True)
            del self.active_traces[tid]
        self.active_spans.clear()

    def force_flush(self):
        # No-op for now; could be used to persist to disk
        pass
