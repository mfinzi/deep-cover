import torch
import numpy as np
from decimal import Decimal, getcontext
import base64
import zlib
import math
import copy
from baseconvert import base
torch.hub.list('pytorch/fairseq')
import fairseq
from fairseq.search import Sampling

# def text2decimal(text):
#     compressed_text_data = text.encode('utf-8')#zlib.compress()
#     message_decimal = Decimal('0.'+base(list(compressed_text_data)[::-1], 256, 10, string=True)[::-1])
#     return message_decimal

def decimal2bits(decimal,bits_encoded):
    #'0.d0d1...' -> (...,d1,d0) -> [...,b1,b0] -> [b0,b1,...,b(bits_encoded-1)]
    # decimals_encoded = math.ceil(bits_encoded*np.log(2)/np.log(10)+1)
    # base10digits = decimal.as_tuple().digits[:decimals_encoded]#[::-1]
    # base2bits = base(base10digits,10,2)#[::-1]
    # return base2bits[:bits_encoded]#base2bits
    output_bits = []
    while len(output_bits)<bits_encoded:
        if decimal > 0.5:
            output_bits.append(1)
            decimal -= Decimal(0.5)
        else:
            output_bits.append(0)
        decimal *=2
    return output_bits

def bits2text(bits):
    # [b0,b1,...] -> [..., b2,b1,b0] -> [..., h2,h1,h0] -> [h0,h1,..]
    base256bytes = bytes(base(bits[::-1],2,256)[::-1])
    return base256bytes

def text2bits(text):
    encoded_data = text.encode('utf-8')
    #[h0,h1,..] -> [hn,hn-1,...h0] -> [..., b2,b1,b0] -> [b0,b1,...]
    return base(list(encoded_data)[::-1],256,2)[::-1]

def bits2decimal(bits):
    #[b0,b1,...] -> [...,b1,b0] -> [...,d1,d0] -> [d0,d1,...] -> '0.d0d1d2...'
    #return Decimal('0.'+base(bits,2,10,string=True))
    #return Decimal('0.'+str(bits2int(bits))[::-1])
    val = Decimal(0)
    for i,bit in enumerate(bits):
        val += bit*Decimal(2**(-i-1))
    return val

def decimals2text(decimals,all_bits_encoded):
    all_bits = []
    for decimal,bits_encoded in zip(decimals,all_bits_encoded):
        all_bits.extend(decimal2bits(decimal,bits_encoded))
    return bits2text(all_bits)


# def decimal2text(decimal,bits_encoded=-1):
#     base10digits = decimal.as_tuple().digits[::-1]
#     base2bits = base(base10digits,10,2)[::-1][:bits_encoded][::-1]
#     base256bytes = bytes(base(base2bits,2,256))[::-1]
#     decompressed_text = base256bytes#zlib.decompress(base256bytes)
#     return bytes(decompressed_text)#.decode('utf-8')

class EncodeSampling(Sampling):

    def __init__(self, tr,tgt_dict,secret_message, encode=True,sampling_topk=-1, sampling_topp=-1.0):
        #self.pad,self.unk,self.eos,self.vocab_size = puev
        super().__init__(tgt_dict, sampling_topk=sampling_topk, sampling_topp=sampling_topp)
        if encode:
            print("was string")
            self.message_val=secret_message#text2decimal(secret_message)
        else:
            print("was not string")
            self.message_val = Decimal(0)
            self.encoded_message = secret_message #actually cover text
        self.encode=encode
        self.tr = tr
        #self.upper = Decimal(1)
        #self.lower = Decimal(0)
        self.width = Decimal(1)
    def step(self, step, lprobs, scores):
        super()._init_buffers(lprobs)
        bsz, beam_size, vocab_size = lprobs.size()

        if step == 0:
            # at the first step all hypotheses are equally likely, so use
            # only the first beam
            lprobs = lprobs[:, ::beam_size, :].contiguous()

        # we exclude the first two vocab items, one of which is pad
        assert self.pad <= 1, 'sampling assumes the first two symbols can be ignored'
        lprobs_nopad = lprobs[:, :, 2:]

        if self.sampling_topp > 0:
            # only sample from the smallest set of words whose cumulative probability mass exceeds p
            probs_nopad, top_indices = self._sample_topp(lprobs_nopad)
        elif self.sampling_topk > 0:
            # only sample from top-k candidates
            lprobs_nopad, top_indices = lprobs_nopad.topk(self.sampling_topk)
            probs_nopad = lprobs_nopad.exp_()
        else:
            probs_nopad = lprobs_nopad.exp_()
        rescaled_probsnpad = probs_nopad/probs_nopad.sum(-1,keepdims=True)
        #print(rescaled_probsnpad[:,:,:5])
        assert beam_size==1, "Only beam size of 1 supported for message encoding right now"
        assert bsz==1, "Further work required to support batch size >1"
        np_probs = rescaled_probsnpad[0,0].cpu().data.numpy()
        cumsum_probs = np.cumsum(np_probs)
        if self.encode:
            # We are trying to encode the hidden message with our choice of tokens
            selected_bin = np.digitize(np.array([float(self.message_val)]),cumsum_probs)[0]
            self.indices_buf = torch.tensor([selected_bin]).long().view(bsz,beam_size).to(probs_nopad.device)
            bin_size = np_probs[selected_bin]
            bin_start= cumsum_probs[selected_bin-1] if selected_bin>0 else 0
            self.message_val = ((self.message_val - Decimal(float(bin_start)))/Decimal(float(bin_size)))#.quantize(self.message_val)
            self.width = self.width*Decimal(float(bin_size))
            #print(bin_size)
            if not step%30: print(-(np.log(float(self.width)))/np.log(2))
        else:
            # We are trying to decode the hidden message by the chosen tokens
            indices_reverse_mapping = dict(zip(top_indices[0,0].cpu().data.numpy(),np.arange(top_indices.shape[-1])))
            selected_bin = indices_reverse_mapping[(self.encoded_message[step]-2).cpu().data.numpy().item()]
            self.indices_buf = torch.tensor([selected_bin]).long().view(bsz,beam_size).to(probs_nopad.device)
            bin_size = np_probs[selected_bin]
            bin_start= cumsum_probs[selected_bin-1] if selected_bin>0 else 0
            self.message_val += Decimal(float(bin_start))*self.width
            self.width = self.width*Decimal(float(bin_size))
            if not step%30: print(-(np.log(float(self.width)))/np.log(2))

        #print(probs_nopad.sum())
        if step == 0:
            # expand to beam size
            probs_nopad = probs_nopad.expand(bsz, beam_size, -1)

        # gather scores
        torch.gather(
            probs_nopad,
            dim=2,
            index=self.indices_buf.unsqueeze(-1),
            out=self.scores_buf,
        )
        self.scores_buf = self.scores_buf.log_().view(bsz, -1)

        # remap indices if using top-k or top-P sampling
        if self.sampling_topk > 0 or self.sampling_topp > 0:
            self.indices_buf = torch.gather(
                top_indices.expand(bsz, beam_size, -1),
                dim=2,
                index=self.indices_buf.unsqueeze(-1),
            ).squeeze(2)

        # remap indices since we excluded the first two vocab items
        self.indices_buf.add_(2)
        #print(self.indices_buf)
        if step == 0:
            self.beams_buf = self.indices_buf.new_zeros(bsz, beam_size)
        else:
            self.beams_buf = torch.arange(0, beam_size, out=self.beams_buf).repeat(bsz, 1)
            # make scores cumulative
            self.scores_buf.add_(
                torch.gather(
                    scores[:, :, step - 1],
                    dim=1,
                    index=self.beams_buf,
                )
            )
        self.tr.bits = int(-(np.log(float(self.width)))/np.log(2))
        if not self.encode: self.tr.vall = self.message_val#+Decimal(self.width/2)
        #if not self.encode: print(self.message_val)#decimal2text(self.message_val))
        return self.scores_buf, self.indices_buf, self.beams_buf



def generate_hidden(self,message, tokens,decode=False, beam = 1, verbose = False,**kwargs):
    sample = self._build_sample(tokens)

    # build generator using current args as well as any kwargs
    gen_args = copy.copy(self.args)
    gen_args.beam = beam
    for k, v in kwargs.items():
        setattr(gen_args, k, v)
    generator = self.task.build_generator(gen_args)
    # Code added here
    #puev = generator.search.pad,generator.search.unk,generator.search.eos,generator.search.vocab_size
    generator.search = EncodeSampling(self,self.task.target_dictionary,message,encode=not decode,
                sampling_topk=getattr(gen_args, 'sampling_topk', -1),
                sampling_topp=getattr(gen_args, 'sampling_topp', -1.0))
    # 
    translations = self.task.inference_step(generator, self.models, sample)

    if verbose:
        src_str_with_unk = self.string(tokens)
        print('S\t{}'.format(src_str_with_unk))

    def getarg(name, default):
        return getattr(gen_args, name, getattr(self.args, name, default))

    # Process top predictions
    hypos = translations[0]
    if verbose:
        for hypo in hypos:
            hypo_str = self.decode(hypo['tokens'])
            print('H\t{}\t{}'.format(hypo['score'], hypo_str))
            print('P\t{}'.format(
                ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
            ))
            if hypo['alignment'] is not None and getarg('print_alignment', False):
                print('A\t{}'.format(
                    ' '.join(map(lambda x: str(utils.item(x)), hypo['alignment'].int().cpu()))
                ))
    if decode:
        #print(self.decode(hypos[0]['tokens']))
        return self.vall,self.bits
    else:
        return hypos,self.bits


def bind(instance, func, as_name):
    setattr(instance, as_name, func.__get__(instance, instance.__class__))


def encode_short_text(reference_text,secret_bits,models,**kwargs):
    getcontext().prec = 500
    en2other, other2en = models
    n = len(en2other.encode(reference_text))
    otherlang_covertext = en2other.translate(reference_text,min_len=4*n//5)
    print(secret_bits)
    hidden_decimal = bits2decimal(secret_bits)
    print(hidden_decimal)
    en_with_hidden,bits_encoded = generate_hidden(other2en,message=hidden_decimal,tokens=other2en.encode(otherlang_covertext),
                                    beam=1,sampling=True,min_len=4.5*n//5,**kwargs)
    payload_text = other2en.decode(en_with_hidden[0]['tokens'])
    return payload_text,bits_encoded

def decode_short_text(reference_text,payload_text,models,**kwargs):
    getcontext().prec = 500
    en2other, other2en = models
    n = len(en2other.encode(reference_text))
    otherlang_covertext = en2other.translate(reference_text,min_len=4*n//5)
    hidden_decimal,bits_decoded = generate_hidden(other2en,message = en2other.encode(payload_text),decode=True, beam=1, sampling=True,
                                tokens=other2en.encode(otherlang_covertext), min_len=4.5*n//5,**kwargs)
    print(hidden_decimal)
    print(bits_decoded)
    decoded_secret_bits = decimal2bits(hidden_decimal,bits_decoded)
    print(decoded_secret_bits)
    return decoded_secret_bits#decimal2text(hidden_decimal)


def encode_long_text(long_reference_text,secret_message,models,**kwargs):
    paragraphs = long_reference_text.split('\n')
    all_payload_text,all_numbits_encoded = [],0
    secret_message_bits  = text2bits(secret_message)
    for paragraph in paragraphs:
        new_payload, new_bits_encoded = encode_short_text(paragraph,secret_message_bits[all_numbits_encoded:],models,**kwargs)
        all_payload_text.append(new_payload)
        #print(secret_message_bits[all_numbits_encoded:all_numbits_encoded+new_bits_encoded])
        all_numbits_encoded+=new_bits_encoded
        print(f"{new_bits_encoded} new bits encoded")
        # use numbits encoded to shift the secret message
    #print("all of the bits",all_numbits_encoded)
    #print("secret bits",secret_message_bits)
    return '\n'.join(all_payload_text), all_numbits_encoded

def decode_long_text(long_reference_text,long_payload_text,models,**kwargs):
    reference_paragraphs = long_reference_text.split('\n')
    payload_paragraphs = long_payload_text.split('\n')
    decoded_bits = []
    for ref_par,payload_par in zip(reference_paragraphs,payload_paragraphs):
        new_bits_decoded = decode_short_text(ref_par,payload_par,models,**kwargs)
        decoded_bits.extend(new_bits_decoded)
        #print(new_bits_decoded)
        print(f"{len(new_bits_decoded)} new bits decoded")
    #print(decoded_bits)
    return bits2text(decoded_bits)