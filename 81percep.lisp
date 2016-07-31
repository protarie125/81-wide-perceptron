(make-random-state t)

(defparameter noi 81) ;; number of inputs
(defparameter noh 81) ;; number of hids
(defparameter noo 81) ;; number of outputs
(defparameter w-ij (make-hash-table :test #'equal)) ;; (i . j) = w-ij
(defparameter v-jk (make-hash-table :test #'equal)) ;; (j . k) = v-jk

(defun rand0 ()
  (/ (- (random 4001) 2000) 4000)) ;; give -0.5 ~ 0.5

(loop
   for i
   below (1+ noi)
   do (loop
	 for j
	 from 1
	 below (1+ noh)
	 do (setf (gethash (cons i j) w-ij) (rand0))))

(loop
   for j
   below (1+ noh)
   do (loop
	 for k
	 from 1
	 below (1+ noo)
	 do (setf (gethash (cons j k) v-jk) (rand0))))

(defun sigmoid (x)
  (if (< x -88)
      0
      (/ 1 (1+ (exp (- x))))))

;;;;; eval output
(defun get-y (x) ;; x0 = 1, x = (x0, x1, ..., x_noi)
  (let ((y (make-list (1+ noh))))
    (setf (nth 0 y) 1) ;; y0 = 1
    (loop
       for j
       from 1
       below (1+ noh)
       do (setf (nth j y)
		(let ((sum-wx (gethash (cons 0 j) w-ij)))
		  (loop
		     for i
		     from 1
		     below (1+ noi)
		     do (incf sum-wx (* (gethash (cons i j) w-ij)
					(nth i x))))
		  (sigmoid sum-wx))))
    y))

(defun get-z (y) ;; y0 = 1, y = (y0, y1, ..., y_noh)
  (let ((z (make-list (1+ noo)))) ;; z0 = nil
    (loop
       for k
       from 1
       below (1+ noo)
       do (setf (nth k z)
		(let ((sum-vy (gethash (cons 0 k) v-jk)))
		  (loop
		     for j
		     from 1
		     below (1+ noh)
		     do (incf sum-vy (* (gethash (cons j k) v-jk)
					(nth j y))))
		  (sigmoid sum-vy))))
    z))

(defun get-output (x)
  (get-z (get-y x)))

;;;;; learn
(defparameter alpha 0.01)

(defun learn-percep (x target)
  (let ((z-k       (get-output x))
	(e-k       (make-hash-table :test #'equal)) ;; (k) = e-k
	(f-prime-k (make-hash-table :test #'equal)) ;; (k) = f-prime-k
	(delta-k   (make-hash-table :test #'equal)) ;; (k) = delta-k
	(dv-jk     (make-hash-table :test #'equal)) ;; (j . k) = dv-jk
	(v-jk-nu   (make-hash-table :test #'equal)) ;; (j . k) = v-jk-nu
	;;;
	(y-j       (get-y x))
	(e-j       (make-hash-table :test #'equal)) ;; (j) = e-j
	(f-prime-j (make-hash-table :test #'equal)) ;; (j) = f-prime-j
	(delta-j   (make-hash-table :test #'equal)) ;; (j) = delta-j
	(dw-ij     (make-hash-table :test #'equal)) ;; (i . j) = dw-ij
	(w-ij-nu   (make-hash-table :test #'equal))) ;; (i . j) = w-ij-nu
    (loop
       for k
       from 1
       below (1+ noo)
       do (progn
	    (setf (gethash (list k) e-k)
		  (- (nth k target)
		     (nth k z-k)))
	    (setf (gethash (list k) f-prime-k)
		  (* (nth k z-k)
		     (- 1 (nth k z-k))))
	    (setf (gethash (list k) delta-k)
		  (* (gethash (list k) f-prime-k)
		     (gethash (list k) e-k)))))
    (loop
       for j
       below (1+ noh)
       do (loop
	     for k
	     from 1
	     below (1+ noo)
	     do (progn
		  (setf (gethash (cons j k) dv-jk)
			(* alpha
			   (nth j y-j)
			   (gethash (list k) delta-k)))
		  (setf (gethash (cons j k) v-jk-nu)
			(+ (gethash (cons j k) v-jk)
			   (gethash (cons j k) dv-jk))))))
    ;;;
    (loop
       for j
       below (1+ noh)
       do (progn
	    (setf (gethash (list j) e-j)
		  (let ((sum-delta-v 0))
		    (loop
		       for k
		       from 1
		       below (1+ noo)
		       do (incf sum-delta-v
				(* (gethash (list k) delta-k)
				   (gethash (cons j k) v-jk))))
		    sum-delta-v))
	    (setf (gethash (list j) f-prime-j)
		  (* (nth j y-j)
		     (- 1 (nth j y-j))))
	    (setf (gethash (list j) delta-j)
		  (* (gethash (list j) f-prime-j)
		     (gethash (list j) e-j)))))
    (loop
       for i
       below (1+ noi)
       do (loop
	     for j
	     from 1
	     below (1+ noh)
	     do (progn
		  (setf (gethash (cons i j) dw-ij)
			(* alpha
			   (nth i x)
			   (gethash (list j) delta-j)))
		  (setf (gethash (cons i j) w-ij-nu)
			(+ (gethash (cons i j) w-ij)
			   (gethash (cons i j) dw-ij))))))
    ;;;
    (loop
       for i
       below (1+ noi)
       do (loop
	     for j
	     from 1
	     below (1+ noh)
	     do (setf (gethash (cons i j) w-ij)
		      (gethash (cons i j) w-ij-nu))))
    (loop
       for j
       below (1+ noh)
       do (loop
	     for k
	     from 1
	     below (1+ noo)
	     do (setf (gethash (cons j k) v-jk)
		      (gethash (cons j k) v-jk-nu))))))

(defun l-p-t (x target times)
  (loop
     for n
     below times
     do (progn
	  (learn-percep x target)
	  (if (= 0 (mod (1+ n) 100))
	      (format t ".")
	      nil)))
  (format t "~%Learn ~A times.~%" times)
  (get-output x))

;;;;; test
(defparameter *test1* '(1 0 0  0  0  0  0  0 0 0
			  0 1  1  1  1  1  1 1 0
			  0 1 -1 -1 -1 -1 -1 1 0
			  0 1 -1  0  0  0 -1 1 0
			  0 1 -1  0  0  0 -1 1 0
			  0 1 -1  0  0  0 -1 1 0
			  0 1 -1 -1 -1 -1 -1 1 0
			  0 1  1  1  1  1  1 1 0
			  0 0  0  0  0  0  0 0 0))
(defparameter *target1* '(nil 1 1 1 1 1 1 1 1 1
			      1 1 1 1 1 1 1 1 1
			      1 1 0 0 0 0 0 1 1
			      1 1 0 0 0 0 0 1 1
			      1 1 0 0 0 0 0 1 1
			      1 1 0 0 0 0 0 1 1
			      1 1 0 0 0 0 0 1 1
			      1 1 1 1 1 1 1 1 1
			      1 1 1 1 1 1 1 1 1))
(defparameter *test2* '(1  0  0  0  0  1 -1  0  0  0
			   0  0  0  0  1 -1  0  0  0
			   0  0  0  0  1 -1  0  0  0
			   0  0  0  0  1 -1  0  0  0
			   1  1  1  1  1 -1  0  0  0
			  -1 -1 -1 -1 -1 -1 -1 -1 -1
			   0  0  0  0 -1  1  1  1  1
			   0  0  0  0 -1  1  0  0  0
			   0  0  0  0 -1  1  0  0  0))
(defparameter *target2* '(nil 1 1 1 1 1 0 0 0 0
			      1 1 1 1 1 0 0 0 0
			      1 1 1 1 1 0 0 0 0
			      1 1 1 1 1 0 0 0 0
			      1 1 1 1 1 0 0 0 0
			      0 0 0 0 0 0 0 0 0
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1))
(defparameter *test3* '(1  0  0  0  0 -1  1  0  0  0
			   0  0  0  0 -1  1  0  0  0
			   0  0  0  0 -1  1  0  0  0
			   0  0  0  0 -1  1  0  0  0
			  -1 -1 -1 -1 -1  1  0  0  0
			   1  1  1  1  1  1  1  1  1
			   0  0  0  0  1 -1 -1 -1 -1
			   0  0  0  0  1 -1  0  0  0
			   0  0  0  0  1 -1  0  0  0))
(defparameter *target3* '(nil 0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      1 1 1 1 1 1 1 1 1
			      1 1 1 1 1 0 0 0 0
			      1 1 1 1 1 0 0 0 0
			      1 1 1 1 1 0 0 0 0))
(defparameter *test4* '(1  1  1  1  1  0  1 -1 -1  0
			  -1 -1 -1  1  0  1 -1  0 -1
			   0  0 -1  1  0  1  1 -1 -1
			   0  0  0 -1  1  0  1  1  1
			   0  0  0 -1  1  0  1  0  0
			  -1 -1 -1 -1  1  1  1  1  1
			   0  0  0 -1 -1 -1 -1 -1 -1
			   0  0  0 -1  0  0  0  0  0
			   0  0  0 -1  0  0  0  0  0))
(defparameter *target4* '(nil 1 1 1 1 1 1 0 0 0
			      0 0 0 1 1 1 0 0 0
			      0 0 0 1 1 1 1 0 0
			      0 0 0 0 1 1 1 1 1
			      0 0 0 0 1 1 1 1 1
			      0 0 0 0 1 1 1 1 1
			      0 0 0 0 0 0 0 0 0
			      0 0 0 0 0 0 0 0 0
			      0 0 0 0 0 0 0 0 0))
(defparameter *test5* '(1 -1 -1 -1  1  1  1 0 0 0
			   0  0 -1 -1 -1  1 0 0 0
			   0  0  0  0 -1  1 1 1 1
			  -1 -1  0  0  0 -1 1 0 0
			   0 -1 -1  0  0 -1 1 1 1
			   0  0 -1 -1 -1  1 1 0 0
			   0  0 -1  0 -1  1 1 0 0
			  -1 -1 -1  0 -1 -1 1 0 0
			   0  0  0  0  0 -1 1 0 0))
(defparameter *target5* '(nil 0 0 0 1 1 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 0 1 1 1
			      0 0 0 0 0 0 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 1 1 1 1
			      0 0 0 0 0 0 1 1 1
			      0 0 0 0 0 0 1 1 1))
(defparameter *test6* '(1  1  0  1 -1 0 -1  1  0  1
			   0  1  1 -1 0 -1  1  1  0
			   1  1  1 -1 0 -1  1  1  1
			  -1 -1 -1 -1 0 -1 -1 -1 -1
			   0  0  0  0 0  0  0  0  0
			  -1 -1 -1 -1 0 -1 -1 -1 -1
			   1  1  1 -1 0 -1  1  1  1
			   0  1  1 -1 0 -1  1  1  0
			   1  0  1 -1 0 -1  1  0  1))
(defparameter *target6* '(nil 1 1 1 0 0 0 1 1 1
			      1 1 1 0 0 0 1 1 1
			      1 1 1 0 0 0 1 1 1
			      0 0 0 0 0 0 0 0 0
			      0 0 0 0 0 0 0 0 0
			      0 0 0 0 0 0 0 0 0
			      1 1 1 0 0 0 1 1 1
			      1 1 1 0 0 0 1 1 1
			      1 1 1 0 0 0 1 1 1))
(defparameter *test7* '(1
			 0 -1  1  0 0  1 -1 -1  0
			 0 -1  1  0 1  1  1 -1  0
			 0 -1  1  0 1 -1 -1  0  0
			 0 -1 -1  1 1 -1  0 -1  0
			-1 -1 -1 -1 1 -1 -1 -1 -1
			 1 -1  1  1 0  1  1  1 -1
			 1  1  1  0 0  0  1  0  1
			 0  0  0  0 0  0  0  0  0
			 0  0  0  0 0  0  0  0  0))
(defparameter *target7* '(nil
			  0 0 1 1 1 1 0 0 0
			  0 0 1 1 1 1 1 0 0
			  0 0 1 1 1 0 0 0 0
			  0 0 0 1 1 0 0 0 0
			  0 0 0 0 1 0 0 0 0
			  1 0 1 1 1 1 1 1 0
			  1 1 1 1 1 1 1 1 1
			  1 1 1 1 1 1 1 1 1
			  1 1 1 1 1 1 1 1 1))
(defparameter *test8* '(1
			 0  0 -1 1 0 0 1 -1  0
			 0  0 -1 1 0 0 1 -1  0
			 0  0 -1 1 0 0 1 -1  0
			 0  0 -1 1 1 1 1 -1  0
			 0  0 -1 1 0 0 1 -1  0
			 0  0 -1 1 0 0 1 -1  0
			-1 -1 -1 1 0 0 1 -1 -1
 			 0  0 -1 1 0 0 1 -1  0
 			 0  0 -1 1 0 0 1 -1  0))
(defparameter *target8* '(nil
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0
			  0 0 0 1 1 1 1 0 0))
(defparameter *test9* '(1
			0 0  0  0  0  0 0 0 0
			0 0  1  1  1  1 1 0 0
			0 0  1 -1 -1 -1 1 1 1
			0 1  1 -1  0 -1 1 0 0
			0 1 -1 -1 -1 -1 1 0 0
			0 1 -1  0  0 -1 1 0 0
			0 1 -1 -1 -1 -1 1 0 0
			0 1  1  1  1  1 1 1 1
			1 0  0  0  0  1 0 0 0))
(defparameter *target9* '(nil
			  1 1 1 1 1 1 1 1 1
			  1 1 1 1 1 1 1 1 1
			  1 1 1 0 0 0 1 1 1
			  1 1 1 0 0 0 1 1 1
			  1 1 0 0 0 0 1 1 1
			  1 1 0 0 0 0 1 1 1
			  1 1 0 0 0 0 1 1 1
			  1 1 1 1 1 1 1 1 1
			  1 1 1 1 1 1 1 1 1))
(defparameter *test10* '(1
			 0  0  0  0  0  0  0  0  0
			 0  0 -1 -1 -1 -1 -1  0  0
			 0  0 -1  1  1  1 -1 -1 -1
			 0 -1 -1  1  0  1 -1  0  0
			 0 -1  1  1  1  1 -1  0  0
			 0 -1  1  0  0  1 -1  0  0
			 0 -1  1  1  1  1 -1  0  0
			 0 -1 -1 -1 -1 -1 -1 -1 -1
			-1  0  0  0  0 -1  0  0  0))
(defparameter *target10* '(nil
			   0 0 0 0 0 0 0 0 0
			   0 0 0 0 0 0 0 0 0
			   0 0 0 1 1 1 0 0 0
			   0 0 0 1 1 1 0 0 0
			   0 0 1 1 1 1 0 0 0
			   0 0 1 1 1 1 0 0 0
			   0 0 1 1 1 1 0 0 0
			   0 0 0 0 0 0 0 0 0
			   0 0 0 0 0 0 0 0 0))

(defun limit-print (output)
  (loop
     for k
     from 1
     below (1+ noo)
     do (progn
	  (format t "~A " (cond ((< (nth k output) (/ 1 3)) 0)
				((< (/ 2 3) (nth k output)) 1)
				(t (nth k output))))
	  (if (= 0 (mod k 9))
	      (format t "~%")))))

(defun lpt150 ()
  (limit-print (l-p-t *test1* *target1* 150))
  (limit-print (l-p-t *test2* *target2* 150))
  (limit-print (l-p-t *test3* *target3* 150))
  (limit-print (l-p-t *test4* *target4* 150))
  (limit-print (l-p-t *test5* *target5* 150))
  (limit-print (l-p-t *test6* *target6* 150))
  (limit-print (l-p-t *test7* *target7* 150))
  (limit-print (l-p-t *test8* *target8* 150))
  (limit-print (l-p-t *test9* *target9* 150))
  (limit-print (l-p-t *test10* *target10* 150))
  (format t "@@@:::!!!:::@@@~%"))
