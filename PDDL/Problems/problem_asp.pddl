(define (problem hanoi-problem)
(:domain hanoi)
(:objects o0 o1 o2 o3 o4 o5 - obj)
(:init 

(p1 o0 o0)
(p1 o0 o1)
(p1 o1 o2)
(p1 o5 o3)
(p1 o4 o4)
(p1 o5 o5)

;val((1,(1,1)),25) val((1,(1,2)),25) val((1,(2,3)),25) val((1,(6,4)),25) val((1,(5,5)),25) val((1,(6,6)),25)

)
(:goal (and 

(p1 o0 o0)
(p1 o0 o3)
(p1 o1 o1)
(p1 o1 o5)
(p1 o2 o2)
(p1 o5 o4)


;val((1,(1,1)),20) val((1,(1,4)),20) val((1,(2,2)),20) val((1,(2,6)),20) val((1,(3,3)),20) val((1,(6,5)),20)

))
)