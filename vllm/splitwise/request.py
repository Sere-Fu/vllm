class SplitwiseRequest:
    def __init__(self, prefill_endpoint, decoding_rank):
        self.prefill_endpoint = prefill_endpoint
        self.decoding_rank = decoding_rank
        # FIXME: determine prefill_rank by endpoint
        if prefill_endpoint:
            port = int(prefill_endpoint.split(':')[-1].split('/')[0])
        self.prefill_rank = port % 10 if prefill_endpoint else None
        self.future = None
