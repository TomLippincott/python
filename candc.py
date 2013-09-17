import nlp
from os.path import expanduser
class CCGParser:
    def __init__(self, models, pos=None, super=None, morph=None, **args):
        """
        Accepts arguments similar to the command-line tool:
        """
        cfg = nlp.config.Main('CCGParser')
        _models = nlp.config.OpPath(cfg, "base", "base directory for all models", expanduser(models))
        if pos:
            _pos = nlp.config.OpPath(cfg, "pos", "base directory for all models", expanduser(pos))
        else:
            _pos = nlp.config.OpPath(cfg, "pos", "base directory for all models", expanduser(models))
        if super:
            _super = nlp.config.OpPath(cfg, "super", "base directory for all models", expanduser(super))
        else:
            _super = nlp.config.OpPath(cfg, "super", "base directory for all models", expanduser(models))
        if morph:
            _morph = nlp.config.OpPath(cfg, "morph", "base directory for all models", expanduser(morph))
        else:
            _morph = nlp.config.OpPath(cfg, "morph", "base directory for all models", expanduser(models))
        parser_cfg = nlp.ccg.ParserConfig(_models)
        pos_cfg = nlp.tagger.POSConfig(_pos)
        super_cfg = nlp.tagger.SuperConfig(_super)
        int_cfg = nlp.ccg.IntegrationConfig()
        decoder_name = nlp.config.OpString(cfg, "decoder", "the parser decoder [deps, derivs, random]", "derivs")

        self.pos = nlp.tagger.POS(pos_cfg)
        self.sent = nlp.Sentence()
        self.integration = nlp.ccg.Integration(int_cfg, super_cfg, parser_cfg, self.sent)
        self.printer = nlp.ccg.PythonPrinter(self.integration.cats)
        self.decoder = nlp.ccg.DecoderFactory(decoder_name.value)
        
    def parse(self, sentence):
        # 250 max
        self.sent.words = sentence.split()
        self.pos.tag(self.sent, nlp.tagger.VITERBI, 5)
        retval = {}
        if self.integration.parse(self.sent, self.decoder, self.printer, True):
            for k in ["beta", "deps", "deriv", "dict_cutoff", "grs", "nparsed", "nsentences", "reason", "success"]:
                retval[k] = getattr(self.printer, k)
            retval["pos"] = [str(x) for x in self.sent.pos]
            retval["lemmas"] = [str(x) for x in self.sent.lemmas]
            retval["tokens"] = [str(x) for x in self.sent.words]
            return retval                

def compat_print(parse):
    for k, v in parse.iteritems():
        print k, v

if __name__ == "__main__":
    p = CCGParser(models="/anfs/bigdisc/tl318/candc_models/models", pos="/anfs/bigdisc/tl318/candc_models/bio_models", super="/anfs/bigdisc/tl318/candc_models/bio_models")
    compat_print(p.parse("This test requires your attention."))
