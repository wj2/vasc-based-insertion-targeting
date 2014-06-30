
class SWCEntry:
    
    def __init__(self, spec_string, cylinder):
        attributes = spec_string.split(' ')
        ident, struct, x, y, z, rad, par = attributes[:7]
        self.cylinder = cylinder
        self.ident = int(ident)
        self.struct = int(struct)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.rad = float(rad)
        self.par = int(par)

    def get_dimmin(self, dim):
        if self.cylinder:
            dimmin = getattr(self, dim) + 0.5
        else:
            dimmin = getattr(self, dim) - self.rad
        return dimmin

    def get_dimmax(self, dim):
        if self.cylinder:
            dimmax = getattr(self, dim) + 0.5
        else:
            dimmax = getattr(self, dim) + self.rad
        return dimmax

    @property
    def xyz(self):
        return (self.x, self.y, self.z)
    
