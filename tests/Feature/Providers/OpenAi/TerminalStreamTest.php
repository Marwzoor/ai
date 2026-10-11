<?php

use Illuminate\Support\Facades\Http;
use Laravel\Ai\Exceptions\StreamErrorException;
use Laravel\Ai\Responses\Data\FinishReason;
use Laravel\Ai\Responses\StreamedAgentResponse;
use Laravel\Ai\Streaming\Events\StreamEnd;
use Tests\Fixtures\Agents\ProviderOptionsWithToolsAgent;

beforeEach(function (): void {
    config([
        'ai.providers.openai.key' => 'test-key',
        'ai.providers.azure.key' => 'test-key',
        'ai.providers.azure.url' => 'https://my-resource.cognitiveservices.azure.com',
        'ai.providers.azure.deployment' => 'gpt-4o',
    ]);
});

test('incomplete responses preserve usage without executing completed tool calls', function (string $provider, string $reason, FinishReason $expected): void {
    $terminal = $this->responseCompleted(42, 128, cachedTokens: 10, reasoningTokens: 24, output: [
        ['type' => 'function_call', 'status' => 'completed', 'id' => 'fc_1', 'call_id' => 'call_1', 'name' => 'FixedNumberGenerator', 'arguments' => '{}'],
    ]);
    $terminal['type'] = 'response.incomplete';
    $terminal['response']['status'] = 'incomplete';
    $terminal['response']['incomplete_details'] = ['reason' => $reason];

    Http::fake(['*' => Http::response($this->ssePayload([
        $this->responseCreated(),
        $this->outputTextDelta('Partial answer'),
        $this->outputTextDone('Partial answer'),
        $this->outputItemAdded('fc_1', 'call_1', 'FixedNumberGenerator'),
        $this->functionCallArgumentsDone('fc_1', '{}'),
        $terminal,
    ]), headers: ['Content-Type' => 'text/event-stream'])]);

    $response = null;
    $stream = (new ProviderOptionsWithToolsAgent)->stream('Hello', provider: $provider);
    $stream->then(function (StreamedAgentResponse $result) use (&$response): void {
        $response = $result;
    });
    $events = iterator_to_array($stream);
    $end = collect($events)->first(fn ($event): bool => $event instanceof StreamEnd);

    expect($end->reason)->toBe($expected->value)
        ->and($end->usage->inputTokens)->toBe(42)
        ->and($end->usage->outputTokens)->toBe(128)
        ->and($end->usage->reasoningTokens)->toBe(24)
        ->and($end->usage->cacheReadInputTokens)->toBe(10)
        ->and($response->text)->toBe('Partial answer')
        ->and($response->steps->last()->finishReason)->toBe($expected)
        ->and($response->toolResults)->toBeEmpty();

    Http::assertSentCount(1);
})->with(['openai', 'azure'])->with([
    'output limit' => ['max_output_tokens', FinishReason::Length],
    'content filter' => ['content_filter', FinishReason::ContentFilter],
    'unknown incomplete reason' => ['unknown', FinishReason::Error],
]);

test('failed responses surface the provider error', function (string $provider): void {
    Http::fake(['*' => Http::response($this->ssePayload([
        $this->responseCreated(),
        ['type' => 'response.failed', 'response' => [
            'id' => 'resp_1', 'status' => 'failed',
            'error' => ['code' => 'server_error', 'message' => 'The response could not be completed.'],
        ]],
    ]), headers: ['Content-Type' => 'text/event-stream'])]);

    $error = null;

    try {
        iterator_to_array((new ProviderOptionsWithToolsAgent)->stream('Hello', provider: $provider));
    } catch (StreamErrorException $exception) {
        $error = $exception->error;
    }

    expect($error)->not->toBeNull()
        ->and($error->type)->toBe('server_error')
        ->and($error->message)->toBe('The response could not be completed.');
})->with(['openai', 'azure']);

test('streams without a terminal response fail instead of completing successfully', function (string $provider, bool $hasPartialOutput): void {
    $events = $hasPartialOutput ? [$this->responseCreated(), $this->outputTextDelta('Partial answer')] : [];

    Http::fake(['*' => Http::response($this->ssePayload($events), headers: ['Content-Type' => 'text/event-stream'])]);

    expect(fn () => iterator_to_array((new ProviderOptionsWithToolsAgent)->stream('Hello', provider: $provider)))
        ->toThrow(StreamErrorException::class);
})->with(['openai', 'azure'])->with([
    'empty stream' => false,
    'partial output' => true,
]);
