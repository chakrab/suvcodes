from agents import Agent, Runner, OpenAIChatCompletionsModel, trace, set_tracing_disabled, set_trace_processors, InputGuardrail, GuardrailFunctionOutput, \
    InputGuardrailTripwireTriggered, TracingProcessor
from agents.tracing.traces import TraceImpl
from agents.tracing.spans import SpanImpl
from agents.tracing import AgentSpanData, FunctionSpanData, GenerationSpanData, ResponseSpanData, HandoffSpanData, CustomSpanData, GuardrailSpanData, \
    TranscriptionSpanData, SpeechSpanData, SpeechGroupSpanData, MCPListToolsSpanData

class LocalTracingProcessor(TracingProcessor):
    def __init__(self):
        self.active_traces = {}
        self.active_spans = {}

    def on_trace_start(self, trace):
        self.active_traces[trace.trace_id] = trace
        self.exportTrace("Trace Started", self.active_traces[trace.trace_id])

    def on_trace_end(self, trace):
        self.exportTrace("Trace Ended", self.active_traces[trace.trace_id])
        del self.active_traces[trace.trace_id]

    def on_span_start(self, span):
        self.active_spans[span.span_id] = span

    def on_span_end(self, span):
        self.exportSpan("%", self.active_spans[span.span_id])
        del self.active_spans[span.span_id]

    def shutdown(self):
        # Clean up resources
        self.active_traces.clear()
        self.active_spans.clear()

    def force_flush(self):
        # Force processing of any queued items
        pass

    def exportTrace(self, txt, item) -> None:
        print(f"{txt}: {item.name}", flush=True)

    def exportSpan(self, txt, item) -> None:
        if isinstance(item.span_data, AgentSpanData):
            print(f"AGENT>> {txt}: {item.span_data.name}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, FunctionSpanData):
            print(f"FUNCTION>> {txt}: {item.span_data.name}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, GenerationSpanData):
            print(f"GENERATION>> {txt}: {item.span_data.model}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, ResponseSpanData):
            print(f"RESPONSE>> {txt}: {item.span_data.Response}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, HandoffSpanData):
            print(f"HANDOFF>> {txt}: {item.span_data.from_agent} -> {item.span_data.to_agent}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, CustomSpanData):
            print(f"CUSTOM>> {txt}: {item.span_data.name}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, GuardrailSpanData):
            print(f"GUARDRAIL>> {txt}: {item.span_data.name} is {item.span_data.triggered}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, TranscriptionSpanData):
            print(f"TRANSCRIPTION>> {txt}: {item.span_data.model}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, SpeechSpanData):
            print(f"SPEECH>> {txt}: {item.span_data.model}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, SpeechGroupSpanData):
            print(f"SPEECHGROUP>> {txt}: {item.span_data.input}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
        elif isinstance(item.span_data, MCPListToolsSpanData):
            print(f"MCP>> {txt}: {item.span_data.server}, Started: {item.started_at}, Ended: {item.ended_at}", flush=True)
