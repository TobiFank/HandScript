import {Card, CardContent} from '@/components/ui/card';
import {Button} from '@/components/ui/button';
import {getStorageUrl} from '@/config/api';
import {type TrainingSample} from '@/services/api';
import {useState} from 'react';
import {AlertCircle, Eye, Pencil, Trash2} from 'lucide-react';
import {Tooltip, TooltipContent, TooltipTrigger} from '@/components/ui/tooltip';

interface TrainingSampleCardProps {
    sample: TrainingSample;
    onEdit: () => void;
    onDelete: () => void;
    isNew?: boolean;
    needsReview?: boolean;
}

export default function TrainingSampleCard({
                                               sample,
                                               onEdit,
                                               onDelete,
                                               isNew = false,
                                               needsReview = false
                                           }: TrainingSampleCardProps) {
    const [showLineDetail, setShowLineDetail] = useState(false);
    const hasLines = Array.isArray(sample.lines) && sample.lines.length > 0;

    return (
        <Card
            className={`relative ${isNew ? 'ring-2 ring-blue-500' : ''} ${needsReview ? 'ring-2 ring-yellow-500 bg-yellow-200' : ''}`}>
            {/* Status Indicators */}
            <div className="absolute top-2 right-2 flex gap-2">
                {needsReview && (
                    <Tooltip>
                        <TooltipTrigger>
                            <div className="bg-yellow-100 text-yellow-700 p-1 rounded-full">
                                <AlertCircle className="w-4 h-4"/>
                            </div>
                        </TooltipTrigger>
                        <TooltipContent>
                            <p>Needs review - Click edit to verify OCR text</p>
                        </TooltipContent>
                    </Tooltip>
                )}
                {isNew && (
                    <div className="bg-blue-500 text-white text-xs px-2 py-1 rounded-full">
                        New
                    </div>
                )}
            </div>

            <CardContent className="p-4">
                <div className="space-y-4">
                    <div className={`relative ${needsReview ? 'mt-8' : 'mt-4'}`}>
                        <img
                            src={getStorageUrl(sample.image_path)}
                            alt="Training sample"
                            className="w-full object-contain"
                        />
                        {showLineDetail && hasLines && (
                            <div className="absolute top-0 left-0 right-0 bottom-0">
                                {sample.lines?.map((line, index) => (
                                    <div
                                        key={index}
                                        className="absolute border-2 border-blue-500/50"
                                        style={{
                                            left: `${line.bbox[0]}%`,
                                            top: `${line.bbox[1]}%`,
                                            width: `${line.bbox[2] - line.bbox[0]}%`,
                                            height: `${line.bbox[3] - line.bbox[1]}%`,
                                        }}
                                    >
                                        <div className="absolute -top-5 left-0 text-xs bg-white px-1 rounded border">
                                            Line {index + 1}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between items-center">
                            <h3 className="text-sm font-medium">Transcription</h3>
                            <div className="flex gap-2">
                                {hasLines && (
                                    <Button
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => setShowLineDetail(!showLineDetail)}
                                    >
                                        <Eye className="w-4 h-4 mr-2"/>
                                        {showLineDetail ? 'Hide Lines' : 'Show Lines'}
                                    </Button>
                                )}
                                <Button
                                    variant={needsReview ? "default" : "ghost"}
                                    size="sm"
                                    onClick={onEdit}
                                    className={needsReview ? "bg-yellow-500 hover:bg-yellow-600 text-white" : ""}
                                >
                                    <Pencil className="w-4 h-4 mr-2"/>
                                    {needsReview ? 'Review' : 'Edit'}
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={onDelete}
                                    className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                >
                                    <Trash2 className="w-4 h-4 mr-2"/>
                                    Delete
                                </Button>
                            </div>
                        </div>

                        {hasLines ? (
                            <div className={`space-y-1 ${needsReview ? 'bg-yellow-200 p-2 rounded' : ''}`}>
                                {sample.lines?.map((line, index) => (
                                    <div
                                        key={index}
                                        className={`text-sm p-2 rounded ${needsReview ? 'bg-yellow-200 border-yellow-400 border' : 'bg-gray-50'}`}
                                    >
                                        {line.text}
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div
                                className={`text-sm p-2 ${needsReview ? 'bg-yellow-200 border-yellow-400 border-2 rounded' : ''}`}>

                                {sample.text}
                            </div>
                        )}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}