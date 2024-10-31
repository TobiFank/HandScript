import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';

interface ImagePreviewModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    imageSrc: string;
}

export default function ImagePreviewModal({
                                              open,
                                              onOpenChange,
                                              imageSrc
                                          }: ImagePreviewModalProps) {
    const [zoom, setZoom] = useState(100);
    const [rotation, setRotation] = useState(0);

    const handleZoomIn = () => {
        setZoom(prev => Math.min(prev + 25, 200));
    };

    const handleZoomOut = () => {
        setZoom(prev => Math.max(prev - 25, 50));
    };

    const handleRotate = () => {
        setRotation(prev => (prev + 90) % 360);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="max-w-4xl w-[90vw] h-[90vh] flex flex-col p-0">
                {/* Controls */}
                <div className="flex items-center justify-between p-4 border-b">
                    <div className="flex items-center space-x-2">
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleZoomOut}
                            disabled={zoom <= 50}
                        >
                            <ZoomOut className="w-4 h-4" />
                        </Button>
                        <span className="text-sm">{zoom}%</span>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleZoomIn}
                            disabled={zoom >= 200}
                        >
                            <ZoomIn className="w-4 h-4" />
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={handleRotate}
                        >
                            <RotateCcw className="w-4 h-4" />
                        </Button>
                    </div>
                </div>

                {/* Image Container */}
                <div className="flex-1 overflow-auto bg-gray-50 p-4">
                    <div className="w-full h-full flex items-center justify-center">
                        <img
                            src={imageSrc}
                            alt="Preview"
                            style={{
                                transform: `scale(${zoom / 100}) rotate(${rotation}deg)`,
                                transformOrigin: 'center',
                                transition: 'transform 0.2s ease',
                                maxWidth: '100%',
                                maxHeight: '100%',
                            }}
                            className="object-contain"
                        />
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}