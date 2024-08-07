class SplitwiseRequest:
    def __init__(self, prefill_endpoint, decoding_rank):
        self.prefill_endpoint = prefill_endpoint
        self.decoding_rank = decoding_rank
        # FIXME: determine prefill_rank by endpoint
        self.prefill_rank = 0 if prefill_endpoint else None
        self.future = None
