(define (problem assembly0)
(:domain assembly)
(:objects o1 o2 o3 o4 o5 - obj); roundnut, squarenut, roundpeg, squarepeg, table
(:init 
(p1 o1 o5)
(p1 o2 o5)
(p1 o3 o5)
(p1 o4 o5)
(p1 o5 o5)
)
(:goal (and 
(p1 o1 o3)
(p1 o2 o4)
(p1 o3 o5)
(p1 o4 o5)
(p1 o5 o5)
))
)