module AutomaticDifferentiation where

-- Automatic Differentiation type
data C = C Double C

valC (C a _) = a
derC (C _ x') = x'

zeroC :: C
zeroC = C 0.0 zeroC

constC :: Double -> C
constC a = C a zeroC

dVarC :: Double -> C
dVarC a = C a (constC 1.0)

-- Only show the value and first derivative, rather than an infinite value
instance Show C where
        show x = "(" ++ (show $ valC x) ++ ", " ++ (show $ valC $ derC x) ++ ")"

instance Eq C where
    (C a x') == (C b y') = a == b

instance Ord C where
    compare (C a _) (C b _) = compare a b

instance Num C where
        (C a x') + (C b y') = C (a + b) (x' + y')
        (C a x') - (C b y') = C (a - b) (x' - y')
        x@(C a x') * y@(C b y') = C (a * b) (x' * y + x * y')
        fromInteger n = constC (fromInteger n)
        abs (C a x') = C (abs a) (abs x')
        signum (C a x') = C (signum a) (signum x')
        
instance Fractional C where
        x@(C a x') / y@(C b y') = C (a / b) ((x' * y - x * y') / (y * y))
        recip (C a x') = C (recip a) (recip x')
        fromRational n = constC (fromRational n)

-- Only implemented functions which are necessary for ANNs
instance Floating C where
        sqrt x@(C a x') = C (sqrt a) (x' / (2 * sqrt x))
        log x@(C a x') = C (log a) (x' / x)
        (**) x@(C a x') y@(C b y') = C (a ** b) (x ** y * (y' * (log x) + (y * x' / x)))
        pi = error "Not implemented"
        exp x@(C a x') = C (exp a) (x' * exp x)
        sin _ = error "Not implemented"
        cos _= error "Not implemented"
        tan _ = error "Not implemented"
        asin _ = error "Not implemented"
        acos _ = error "Not implemented"
        atan _ = error "Not implemented"
        sinh _ = error "Not implemented"
        cosh _ = error "Not implemented"
        tanh x@(C a x') = C (tanh a) (x' * (1 - (tanh x) ** 2))
        asinh _ = error "Not implemented"
        acosh _ = error "Not implemented"
        atanh _ = error "Not implemented"
        logBase _ _ = error "Not implemented"

